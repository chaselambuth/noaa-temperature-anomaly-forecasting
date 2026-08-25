// static/main.js
// This file runs in the browser.
// It reads inputs from index.html, calls your Flask backend at POST /run,
// then displays metrics + plots truth vs prediction using Chart.js.

let chart = null;
const EXPERIMENT_CONFIG = window.EXPERIMENT_CONFIG || {};
const EXPERIMENT_BEST_BY_HORIZON = window.EXPERIMENT_BEST_BY_HORIZON || {};

function parseLags(str, { allowZero = false } = {}) {
  // "0,1,3,6,12" -> [0, 1, 3, 6, 12] when allowZero is true.
  return str
    .split(",")
    .map(s => s.trim())
    .filter(Boolean)
    .map(Number)
    .filter(n => Number.isFinite(n) && (allowZero ? n >= 0 : n > 0));
}

function setStatus(msg) {
  document.getElementById("status").textContent = msg || "";
}

function prettyJSON(obj) {
  return JSON.stringify(obj, null, 2);
}

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;"
  }[ch]));
}

function formatMetric(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number" && Number.isFinite(value)) return value.toFixed(digits);
  return String(value);
}

function displayRunId(runId) {
  if (!runId) return "—";
  const clean = String(runId).replace(/\\/g, "/");
  return clean.split("/").filter(Boolean).pop() || clean;
}

function displayMode(mode) {
  return {
    climatology_residual: "Climatology residual",
    seasonal_ensemble: "Seasonal ensemble",
    ensemble: "Ensemble",
    single: "Single model",
  }[mode] || mode || "—";
}

function csvFromList(values) {
  return Array.isArray(values) ? values.join(",") : "";
}

function recommendationForHorizon(horizon) {
  return EXPERIMENT_BEST_BY_HORIZON[String(horizon)] || EXPERIMENT_BEST_BY_HORIZON["1"] || null;
}

function modelParamsFor(modelName) {
  const params = { ...(EXPERIMENT_CONFIG.model_params || {}) };
  const key = String(modelName || "").toLowerCase();

  if (key === "lstm" || key === "gru") {
    params.torch_epochs = 150;
    params.torch_lr = 0.001;
    params.torch_seq_hidden_size = 24;
    params.torch_seq_num_layers = 1;
    params.torch_seq_dropout = 0.0;
  } else if (key === "cnn1d") {
    params.torch_epochs = 200;
    params.torch_lr = 0.0007;
    params.torch_cnn_channels = 32;
    params.torch_cnn_kernel_size = 5;
  } else {
    params.torch_epochs = 200;
    params.torch_lr = 0.001;
    params.torch_hidden1 = 64;
    params.torch_hidden2 = 32;
    params.torch_dropout = 0.1;
  }

  return params;
}

function activePrimaryModel(mode, singleModel, ensembleModels, seasonalEnsembleModels) {
  if (mode === "single") return singleModel;
  if (mode === "ensemble" && ensembleModels.length === 1) return ensembleModels[0];
  if (mode === "seasonal_ensemble") {
    const unique = new Set(Object.values(seasonalEnsembleModels).flat());
    if (unique.size === 1) return Array.from(unique)[0];
    if (unique.has("lstm")) return "lstm";
    if (unique.has("gru")) return "gru";
    if (unique.has("cnn1d")) return "cnn1d";
  }
  return singleModel;
}

function setChecked(id, value) {
  const el = document.getElementById(id);
  if (el) el.checked = Boolean(value);
}

function updateEnsoVisibility() {
  const useEnso = document.getElementById("use_enso").checked;
  const ensoOptions = document.getElementById("ensoOptions");
  if (ensoOptions) ensoOptions.style.display = useEnso ? "grid" : "none";
}

function updateRecommendationText() {
  const horizon = Number(document.getElementById("horizon").value || EXPERIMENT_CONFIG.horizon || 1);
  const rec = recommendationForHorizon(horizon);
  const summary = document.getElementById("recommendation_summary");
  const detail = document.getElementById("recommendation_detail");

  if (!rec) {
    if (summary) summary.value = "Notebook 06 recommended setup";
    if (detail) detail.textContent = "";
    return;
  }

  if (summary) summary.value = rec.label || "Notebook 06 recommended setup";
  if (detail) {
    detail.textContent = rec.test_rmse
      ? `Notebook 06 test RMSE ${formatMetric(rec.test_rmse)} | MAE ${formatMetric(rec.test_mae)} | bias ${formatMetric(rec.test_bias)}`
      : "Notebook 06 single-model residual recommendation";
    if (rec.note) detail.textContent += ` | ${rec.note}`;
  }
}

function applyNotebookRecommendation() {
  const horizon = Number(document.getElementById("horizon").value || EXPERIMENT_CONFIG.horizon || 1);
  const rec = recommendationForHorizon(horizon);
  const cfg = EXPERIMENT_CONFIG;

  document.getElementById("mode").value = cfg.mode || "single";
  document.getElementById("model_name").value = cfg.model_name || "lstm";

  document.querySelectorAll(".seasonal-ensemble-model").forEach((el) => {
    const models = cfg.seasonal_ensemble_models?.[el.dataset.group] || [];
    el.checked = models.includes(el.value);
  });

  document.getElementById("K").value = cfg.K ?? 10;
  document.getElementById("pc_lags").value = csvFromList(cfg.pc_lags) || "1,2,3,6,12";
  document.getElementById("y_lags").value = csvFromList(cfg.y_lags) || "1,2,3,6,12";
  document.getElementById("sst_pc_lags").value = csvFromList(cfg.sst_pc_lags) || "0,1,3,6,12";
  document.getElementById("ar_lags").value = csvFromList(cfg.ar_lags) || "1,2,3,6,12,18,24";
  document.getElementById("half_life_months").value = cfg.half_life_months ?? 48;
  document.getElementById("climatology_years").value = cfg.climatology_years ?? 18;
  document.getElementById("enso_lags").value = csvFromList(cfg.enso_lags) || "0";

  setChecked("use_sst", cfg.use_sst);
  setChecked("use_teleconnections", cfg.use_teleconnections);
  setChecked("use_qmap", cfg.use_qmap);
  setChecked("residual_deseasonalize", cfg.residual_deseasonalize);
  setChecked("use_enso", cfg.use_enso);
  setChecked("use_climatology_residual", cfg.use_climatology_residual);

  updateModeVisibility();
  updateEnsoVisibility();
  updateRecommendationText();
}

function seasonalGroupLabel(group) {
  return {
    JFM: "January-March",
    AMJ: "April-June",
    JAS: "July-September",
    OND: "October-December",
  }[group] || group;
}

function metricRows(title, metrics) {
  if (!metrics || Object.keys(metrics).length === 0) {
    return `
      <section class="metrics-block">
        <h5>${title}</h5>
        <p class="metrics-empty">No ${title.toLowerCase()} available for this run.</p>
      </section>
    `;
  }

  const labels = [
    ["RMSE", "RMSE"],
    ["MAE", "MAE"],
    ["Bias", "Bias"],
    ["Acc@0.25", "Accuracy @ 0.25"],
    ["Acc@0.35", "Accuracy @ 0.35"],
    ["Acc@0.50", "Accuracy @ 0.50"],
  ];

  const rows = labels
    .filter(([key]) => metrics[key] !== undefined)
    .map(([key, label]) => `<tr><td>${label}</td><td>${formatMetric(metrics[key])}</td></tr>`)
    .join("");

  return `
    <section class="metrics-block">
      <h5>${title}</h5>
      <table class="metric-table">
        <tbody>${rows}</tbody>
      </table>
    </section>
  `;
}

function friendlyErrorMessage(error) {
  const detail = String(error || "Unknown error");

  if (/FileNotFoundError|No such file|cannot find|not found/i.test(detail)) {
    return {
      title: "Data file not found",
      body: "Open Advanced Options and check that the NOAA files are in noaa_raw_data/ or update the path fields to match your computer.",
      detail
    };
  }

  if (/torch|paging file|WinError 1455|out of memory/i.test(detail)) {
    return {
      title: "Model library could not load",
      body: "The app reached the Python model stack but could not load a required library. Closing memory-heavy apps or using the Docker setup can help.",
      detail
    };
  }

  return {
    title: "Forecast run failed",
    body: "The backend returned an error while preparing or running the forecast.",
    detail
  };
}

function renderMetrics(data) {
  const metricsEl = document.getElementById("metrics");

  if (data.error) {
    const friendly = friendlyErrorMessage(data.error);
    metricsEl.innerHTML = `
      <div class="metrics-error">
        <strong>${escapeHTML(friendly.title)}</strong>
        <p>${escapeHTML(friendly.body)}</p>
        <code class="error-detail">${escapeHTML(friendly.detail)}</code>
      </div>
    `;
    return;
  }

  const selectedModel = data.mode === "climatology_residual"
    ? data.selected_models?.join(", ") || "Residual model"
    : data.seasonal_ensemble_models && Object.keys(data.seasonal_ensemble_models).length
    ? "Seasonal ensemble"
    : data.selected_models?.length
    ? `Ensemble: ${data.selected_models.join(", ")}`
    : data.selected_model || data.summary?.best_overall_model || "Run complete";
  const latestTarget = data.latest_forecast?.target_month || "—";
  const latestPred = formatMetric(data.latest_forecast?.predicted_anomaly);
  const runId = displayRunId(data.summary?.run_id);
  const bestModel = data.mode === "climatology_residual"
    ? `${data.config?.climatology_years ?? 18}-year climatology residual`
    : data.summary?.best_overall_model || "—";
  const monthChoice = data.summary?.used_month_choice ? "Yes" : "No";

  metricsEl.innerHTML = `
    <section class="metrics-top">
      <div class="metrics-hero">
        <h5>Run Summary</h5>
        <p class="hero-model">${selectedModel}</p>
        <p class="hero-sub">Best overall model: ${bestModel}</p>
      </div>
      <div class="metrics-kpis">
        <div class="kpi">
          <div class="kpi-label">Latest Forecast (°C anomaly)</div>
          <div class="kpi-value">${latestPred}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Target Month</div>
          <div class="kpi-value">${latestTarget}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Month Choice</div>
          <div class="kpi-value">${monthChoice}</div>
        </div>
      </div>
    </section>
    ${metricRows("Test Metrics", data.test_metrics)}
    ${metricRows("Validation Metrics", data.val_metrics)}
    <section class="metrics-block">
      <h5>Run Details</h5>
      <table class="metric-table">
        <tbody>
          <tr><td>Run ID</td><td>${runId}</td></tr>
          <tr><td>Horizon</td><td>${data.horizon ?? "—"}</td></tr>
          <tr><td>Mode</td><td>${displayMode(data.mode)}</td></tr>
        </tbody>
      </table>
    </section>
  `;
}

function renderChart(labels, truth, pred) {
  const ctx = document.getElementById("chart").getContext("2d");

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Truth (anomaly)",
          data: truth,
          borderColor: "#21313a",
          backgroundColor: "rgba(33, 49, 58, 0.08)",
          tension: 0.15,
          pointRadius: 2,
          borderWidth: 2
        },
        {
          label: "Prediction",
          data: pred,
          borderColor: "#167f87",
          backgroundColor: "rgba(22, 127, 135, 0.10)",
          tension: 0.15,
          pointRadius: 2,
          borderWidth: 3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "top",
          labels: {
            usePointStyle: true,
            boxWidth: 8,
            color: "#21313a",
            font: { weight: "600" }
          }
        },
        tooltip: { enabled: true }
      },
      scales: {
        x: {
          grid: { color: "rgba(33, 49, 58, 0.06)" },
          ticks: { color: "#64737c", maxRotation: 0, autoSkip: true, maxTicksLimit: 10 }
        },
        y: {
          grid: { color: "rgba(33, 49, 58, 0.08)" },
          ticks: { color: "#64737c" },
          title: { display: true, text: "Temperature anomaly (°C)", color: "#64737c" }
        }
      }
    }
  });
}

function getSelectedEnsembleModels() {
  return Array.from(document.querySelectorAll(".ensemble-model:checked")).map((el) => el.value);
}

function getSelectedSeasonalEnsembleModels() {
  const groups = { JFM: [], AMJ: [], JAS: [], OND: [] };
  document.querySelectorAll(".seasonal-ensemble-model:checked").forEach((el) => {
    const group = el.dataset.group;
    if (group && groups[group]) groups[group].push(el.value);
  });
  return groups;
}

function updateModeVisibility() {
  const mode = document.getElementById("mode").value;
  document.getElementById("singleModelSection").style.display = mode === "single" ? "block" : "none";
  document.getElementById("ensembleSection").style.display = mode === "ensemble" ? "block" : "none";
  document.getElementById("seasonalEnsembleSection").style.display = mode === "seasonal_ensemble" ? "block" : "none";
}

async function runExperiment() {
  // 1) Read values from HTML inputs
  const mode = document.getElementById("mode").value;
  const horizon = Number(document.getElementById("horizon").value);
  const model_name = document.getElementById("model_name").value;
  const selectedEnsembleModels = getSelectedEnsembleModels();
  const selectedSeasonalEnsembleModels = getSelectedSeasonalEnsembleModels();
  const ensemble_models = mode === "ensemble" ? selectedEnsembleModels : [];
  const seasonal_ensemble_models = mode === "seasonal_ensemble" ? selectedSeasonalEnsembleModels : {};
  const K = Number(document.getElementById("K").value);
  const pc_lags = parseLags(document.getElementById("pc_lags").value);
  const y_lags = parseLags(document.getElementById("y_lags").value);
  const sst_pc_lags = parseLags(document.getElementById("sst_pc_lags").value);
  const ar_lags = parseLags(document.getElementById("ar_lags").value);
  const enso_lags = parseLags(document.getElementById("enso_lags").value, { allowZero: true });
  const half_life_months = Number(document.getElementById("half_life_months").value);
  const climatology_years = Number(document.getElementById("climatology_years").value);

  const tavg_file = document.getElementById("tavg_file").value.trim();
  const sst_dir = document.getElementById("sst_dir").value.trim();
  const enso_csv = document.getElementById("enso_csv").value.trim();
  const runs_dir = document.getElementById("runs_dir").value.trim();
  const use_sst = document.getElementById("use_sst").checked;
  const use_enso = document.getElementById("use_enso").checked;
  const use_teleconnections = document.getElementById("use_teleconnections").checked;
  const use_qmap = document.getElementById("use_qmap").checked;
  const residual_deseasonalize = document.getElementById("residual_deseasonalize").checked;
  const use_climatology_residual = document.getElementById("use_climatology_residual").checked;
  const primaryModelForParams = activePrimaryModel(mode, model_name, ensemble_models, seasonal_ensemble_models);

  // Basic validation to avoid confusing backend errors
  if (!Number.isFinite(horizon) || horizon <= 0) return alert("horizon must be a positive number.");
  if (!Number.isFinite(K) || K <= 0) return alert("K must be a positive number.");
  if (mode === "ensemble" && ensemble_models.length === 0) return alert("Select at least one model for the ensemble.");
  if (ensemble_models.length > 5) return alert("You can select up to 5 ensemble models.");
  if (mode === "seasonal_ensemble") {
    for (const [group, models] of Object.entries(seasonal_ensemble_models)) {
      if (models.length === 0) return alert(`Select at least one model for ${seasonalGroupLabel(group)}.`);
      if (models.length > 5) return alert(`You can select up to 5 models for ${seasonalGroupLabel(group)}.`);
    }
  }
  if (!pc_lags.length) return alert("Please enter at least one PC lag.");
  if (!y_lags.length) return alert("Please enter at least one Y lag.");
  if (!sst_pc_lags.length) return alert("Please enter at least one SST PC lag.");
  if (!ar_lags.length) return alert("Please enter at least one AR lag.");
  if (use_enso && !enso_lags.length) return alert("Please enter at least one ENSO lag, or turn ENSO off.");
  if (!Number.isFinite(half_life_months) || half_life_months <= 0) return alert("half_life_months must be a positive number.");
  if (!Number.isFinite(climatology_years) || climatology_years <= 0) return alert("climatology_years must be a positive number.");
  if (!tavg_file) return alert("Please provide a tavg_file path.");
  if (!sst_dir) return alert("Please provide an sst_dir path.");
  if (use_enso && !enso_csv) return alert("Please provide an ENSO CSV path, or turn ENSO off.");

  const payload = {
    mode,
    model_name,
    ensemble_models,
    seasonal_ensemble_models,
    ...modelParamsFor(primaryModelForParams),
    horizon,
    K,
    pc_lags,
    y_lags,
    sst_pc_lags,
    ar_lags,
    enso_lags,
    half_life_months,
    climatology_years,
    tavg_file,
    sst_dir,
    enso_csv,
    runs_dir,
    use_enso,
    use_sst,
    use_teleconnections,
    use_qmap,
    residual_deseasonalize,
    use_climatology_residual,
    deep_internal_val_months: EXPERIMENT_CONFIG.deep_internal_val_months ?? 24,
    month_group_size: EXPERIMENT_CONFIG.month_group_size ?? 12
  };

  // 2) Call Flask backend
  setStatus("Please wait. This may take a moment to run; more complex architectures can take longer.");
  document.getElementById("runBtn").disabled = true;

  let res;
  try {
    res = await fetch("/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
  } catch (err) {
    console.error(err);
    setStatus("");
    document.getElementById("runBtn").disabled = false;
    return alert("Network error calling backend. Is Flask running?");
  }

  // 3) Handle backend errors cleanly
  if (!res.ok) {
    const data = await res.json().catch(async () => ({ error: await res.text() }));
    const friendly = friendlyErrorMessage(data.error);
    console.error("Backend error:", data);
    setStatus("");
    document.getElementById("runBtn").disabled = false;
    renderMetrics(data);
    return alert(`Forecast run failed (${res.status}).\n\n${friendly.title}\n${friendly.body}`);
  }

  // 4) Read JSON response
  const data = await res.json();

  setStatus("Done.");
  document.getElementById("runBtn").disabled = false;

  // 5) Show metrics
  renderMetrics(data);

  // 6) Plot series if present
  if (data.series && data.series.dates) {
    const dates = data.series.dates;
    const y_true = data.series.y_true;
    const y_pred = data.series.y_pred;
    renderChart(dates, y_true, y_pred);
  } else {
    console.warn("No series returned for plotting. Response:", data);
  }
}

// Wire up the button click
document.getElementById("runBtn").addEventListener("click", runExperiment);
document.getElementById("mode").addEventListener("change", updateModeVisibility);
document.getElementById("horizon").addEventListener("change", updateRecommendationText);
document.getElementById("horizon").addEventListener("input", updateRecommendationText);
document.getElementById("use_enso").addEventListener("change", updateEnsoVisibility);
document.getElementById("applyRecommendationBtn").addEventListener("click", applyNotebookRecommendation);
document.querySelectorAll(".ensemble-model").forEach((checkbox) => {
  checkbox.addEventListener("change", (event) => {
    const selected = getSelectedEnsembleModels();
    if (selected.length > 5) {
      event.target.checked = false;
      alert("You can select up to 5 models in the ensemble.");
    }
  });
});
document.querySelectorAll(".seasonal-ensemble-model").forEach((checkbox) => {
  checkbox.addEventListener("change", (event) => {
    const group = event.target.dataset.group;
    const selectedInGroup = Array.from(
      document.querySelectorAll(`.seasonal-ensemble-model[data-group="${group}"]:checked`)
    );
    if (selectedInGroup.length > 5) {
      event.target.checked = false;
      alert(`You can select up to 5 models for ${seasonalGroupLabel(group)}.`);
    }
  });
});
applyNotebookRecommendation();
