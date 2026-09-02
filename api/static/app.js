const logEl = document.getElementById("log");
const jobStatus = document.getElementById("job-status");
const runBtn = document.getElementById("run-btn");
const stopBtn = document.getElementById("stop-btn");
const resultBody = document.getElementById("result-body");
const downloads = document.getElementById("downloads");
const resultSource = document.getElementById("result-source");
const plotWrap = document.getElementById("plot-wrap");
const plot = document.getElementById("plot");
const predictOut = document.getElementById("predict-out");

let cursor = 0;
let lastStatus = "idle";
let gpuState = null;

function selectedDevice() {
  return document.querySelector("input[name=device]:checked").value;
}

function selectedFrameworks() {
  return [...document.querySelectorAll("input[name=fw]:checked")].map((el) => el.value);
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function renderGpu(info) {
  const name = document.getElementById("gpu-name");
  const meta = document.getElementById("gpu-meta");
  const dot = document.getElementById("gpu-dot");
  if (info.torch_cuda) {
    name.textContent = info.torch_device || info.name || "CUDA GPU";
    meta.textContent = `PyTorch CUDA ready · driver ${info.driver || "?"} · CUDA ${info.cuda_driver || "?"}`;
    dot.className = "dot ok";
    return;
  }
  if (info.smi_ok) {
    name.textContent = info.name || "NVIDIA GPU";
    meta.textContent = info.torch_note || `Driver ${info.driver} sees the card; PyTorch CUDA is off`;
    dot.className = "dot warn";
    return;
  }
  name.textContent = "CPU only";
  meta.textContent = "nvidia-smi was not found. CPU runs still work.";
  dot.className = "dot";
}

function renderDownloads(artifacts) {
  const items = [
    ["csv", "results.csv"],
    ["report", "PERFORMANCE_REPORT.md"],
    ["plot", "throughput.png"],
    ["json", "results.json"],
    ["zip", "all results (.zip)"],
  ];
  downloads.innerHTML = items
    .map(([key, label]) => {
      const ready = key === "zip" ? artifacts.csv || artifacts.report || artifacts.plot : artifacts[key];
      const cls = ready ? "" : "off";
      return `<a class="${cls}" href="/api/download/${key}">${label}</a>`;
    })
    .join("");
}

function renderTable(payload) {
  resultSource.textContent = payload.source === "measured" ? "this machine" : "sample numbers";
  const rows = payload.rows || [];
  if (!rows.length) {
    resultBody.innerHTML = `<tr><td colspan="5">Run the pipeline to fill this table.</td></tr>`;
    return;
  }
  resultBody.innerHTML = rows
    .map((row) => {
      const latency = Number(row.latency_ms || 0).toFixed(3);
      const thr = Number(row.throughput_img_s || 0).toFixed(1);
      return `<tr>
        <td>${row.framework || ""}</td>
        <td>${row.device || ""}</td>
        <td>${row.phase || ""}</td>
        <td>${latency}</td>
        <td>${thr}</td>
      </tr>`;
    })
    .join("");
}

function setJobUi(status, extra) {
  lastStatus = status;
  const running = status === "running";
  runBtn.disabled = running;
  stopBtn.disabled = !running;
  const labels = {
    idle: "Idle. Ready to run.",
    running: "Running… logs stream below.",
    succeeded: "Finished. Download the files on the right.",
    failed: extra || "Run failed. Check the log.",
  };
  jobStatus.textContent = labels[status] || status;
}

async function refreshStatus() {
  const data = await fetchJson("/api/status");
  gpuState = data.gpu;
  renderGpu(data.gpu);
  renderDownloads(data.artifacts);
  setJobUi(data.job.status, data.job.error);
  if (data.artifacts.plot) {
    plotWrap.hidden = false;
    plot.src = `/api/download/plot?t=${Date.now()}`;
  }
  return data;
}

async function refreshResults() {
  const payload = await fetchJson("/api/results");
  renderTable(payload);
  renderDownloads(payload.artifacts);
  if (payload.artifacts.plot) {
    plotWrap.hidden = false;
    plot.src = `/api/download/plot?t=${Date.now()}`;
  }
}

async function pollLogs() {
  const data = await fetchJson(`/api/logs?cursor=${cursor}`);
  cursor = data.cursor;
  if (data.lines.length) {
    logEl.textContent += (logEl.textContent ? "\n" : "") + data.lines.join("\n");
    logEl.scrollTop = logEl.scrollHeight;
  }
  if (data.status !== lastStatus) {
    setJobUi(data.status);
    if (data.status === "succeeded") {
      await refreshResults();
      await refreshStatus();
    }
  }
}

runBtn.addEventListener("click", async () => {
  try {
    const frameworks = selectedFrameworks();
    if (!frameworks.length) {
      jobStatus.textContent = "Select at least one stage.";
      return;
    }
    if (selectedDevice() === "cuda" && gpuState && !gpuState.torch_cuda) {
      const go = window.confirm(
        "This Python environment cannot use CUDA yet. Continue anyway? The job may stay on CPU or fail."
      );
      if (!go) return;
    }
    cursor = 0;
    logEl.textContent = "";
    await fetchJson("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        device: selectedDevice(),
        frameworks,
        epochs: Number(document.getElementById("epochs").value),
        batch_size: Number(document.getElementById("batch").value),
        max_samples: Number(document.getElementById("samples").value),
        benchmark_iterations: Number(document.getElementById("iters").value),
        warmup: 1,
        synthetic: document.getElementById("synthetic").checked,
        skip_train: document.getElementById("skip-train").checked,
      }),
    });
    setJobUi("running");
  } catch (err) {
    jobStatus.textContent = err.message;
  }
});

stopBtn.addEventListener("click", async () => {
  try {
    await fetchJson("/api/stop", { method: "POST" });
  } catch (err) {
    jobStatus.textContent = err.message;
  }
});

document.getElementById("clear-log").addEventListener("click", () => {
  logEl.textContent = "";
});

document.getElementById("predict-btn").addEventListener("click", async () => {
  const file = document.getElementById("image").files[0];
  if (!file) {
    predictOut.textContent = "Choose an image first.";
    return;
  }
  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch("/predict", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) {
      predictOut.textContent = data.detail || "Prediction failed";
      return;
    }
    const pct = Math.round(data.confidence * 100);
    predictOut.innerHTML = `<strong>${data.label}</strong> · ${pct}% confidence
      <div class="bar"><span style="width:${pct}%"></span></div>`;
  } catch (err) {
    predictOut.textContent = err.message;
  }
});

document.getElementById("image").addEventListener("change", (event) => {
  const file = event.target.files[0];
  document.getElementById("drop-label").textContent = file ? file.name : "Drop an image or click to choose";
});

refreshStatus().then(refreshResults).catch((err) => {
  jobStatus.textContent = err.message;
});
setInterval(() => {
  pollLogs().catch(() => {});
}, 700);
