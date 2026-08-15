import { DEFAULT_IMAGE_SETTINGS, normalizeImageSettings } from "./shared.js";

const enabled = document.querySelector("#enabled");
const minSourceSize = document.querySelector("#min-source-size");
const threshold = document.querySelector("#threshold");
const smartPositioning = document.querySelector("#smart-positioning");
const cssBackgrounds = document.querySelector("#css-backgrounds");
const theme = document.querySelector("#theme");
const warning = document.querySelector("#small-image-warning");
const saved = document.querySelector("#saved");
const filterSite = document.querySelector("#filter-site");
const stateLogo = document.querySelector("#state-logo");
const stateSymbol = document.querySelector("#state-symbol");
const stateIcon = document.querySelector("#state-icon");
const stateCopy = document.querySelector("#state-copy");
let settings = DEFAULT_IMAGE_SETTINGS;
let excludedSites = [];
let currentHost;
let stateTimer;

function renderDetectorState({ enabled, warmingUp }) {
  clearInterval(stateTimer);
  stateTimer = undefined;
  if (warmingUp) {
    stateIcon.classList.add("loading");
    stateCopy.textContent = "Loading local model…";
    let frame = 0;
    stateLogo.hidden = true;
    stateSymbol.hidden = false;
    stateSymbol.src = `icons/loading-${frame}-128.png`;
    stateTimer = setInterval(() => {
      frame = (frame + 1) % 16;
      stateSymbol.src = `icons/loading-${frame}-128.png`;
    }, 125);
    return;
  }
  stateIcon.classList.remove("loading");
  stateCopy.textContent = "Local AI image detector";
  stateSymbol.hidden = true;
  stateLogo.hidden = false;
  stateLogo.src = `icons/${enabled ? "on" : "off"}-128.png`;
  stateLogo.alt = `Cyclopes is ${enabled ? "on" : "off"}`;
}

function rememberSections(openSections) {
  for (const section of document.querySelectorAll("details[id]")) {
    section.open = openSections.includes(section.id);
    section.addEventListener("toggle", () => chrome.storage.local.set({
      openSections: [...document.querySelectorAll("details[id][open]")].map(({ id }) => id),
    }));
  }
}

function renderSettings() {
  document.querySelector("#threshold-value").value = `${threshold.value}%`;
  document.querySelector("#min-source-size-value").value = `${minSourceSize.value} px`;
  warning.hidden = Number(minSourceSize.value) >= DEFAULT_IMAGE_SETTINGS.minSourceSize;
  document.documentElement.dataset.theme = theme.value;
}

async function saveSettings() {
  settings = normalizeImageSettings({
    ...settings,
    minSourceSize: minSourceSize.value,
    threshold: Number(threshold.value) / 100,
    smartPositioning: smartPositioning.checked,
    cssBackgrounds: cssBackgrounds.checked,
    theme: theme.value,
  });
  await chrome.storage.local.set({ enabled: enabled.checked, ...settings });
  saved.value = "Saved";
  setTimeout(() => { saved.value = ""; }, 900);
}

function renderReports(reports) {
  const section = document.querySelector("#reports");
  section.hidden = reports.length === 0;
  if (!reports.length) return;
  document.querySelector("#report-count").textContent = reports.length;
  const grid = document.querySelector("#report-grid");
  grid.replaceChildren();
  for (const report of reports.slice(0, 10)) {
    const link = document.createElement("a");
    link.href = report.source;
    link.target = "_blank";
    link.title = `${report.label === "ai" ? "AI" : "Not AI"} · ${report.site}`;
    const image = document.createElement("img");
    image.src = report.thumbnail || report.source;
    image.alt = "";
    const label = document.createElement("b");
    label.textContent = report.label === "ai" ? "AI" : "REAL";
    link.append(image, label);
    grid.append(link);
  }
}

function renderExcludedSites() {
  const list = document.querySelector("#excluded-sites-list");
  list.replaceChildren();
  document.querySelector("#no-excluded-sites").hidden = excludedSites.length > 0;
  for (const host of excludedSites) {
    const row = document.createElement("div");
    row.className = "excluded-site";
    const name = document.createElement("span");
    name.textContent = host;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove";
    remove.addEventListener("click", async () => {
      excludedSites = excludedSites.filter((site) => site !== host);
      await chrome.storage.local.set({ excludedSites });
      renderExcludedSites();
    });
    row.append(name, remove);
    list.append(row);
  }
  if (currentHost) filterSite.checked = !excludedSites.includes(currentHost);
}

function renderRoute() {
  const sites = location.hash === "#sites";
  document.querySelector("#settings-view").hidden = sites;
  document.querySelector("#sites-view").hidden = !sites;
}

Promise.all([
  chrome.storage.local.get({ enabled: false, excludedSites: [], feedbackReports: [], openSections: [], ...DEFAULT_IMAGE_SETTINGS }),
  chrome.tabs.query({ active: true, currentWindow: true }),
]).then(([values, [tab]]) => {
  settings = normalizeImageSettings(values);
  excludedSites = values.excludedSites;
  enabled.checked = values.enabled;
  minSourceSize.value = settings.minSourceSize;
  threshold.value = Math.round(settings.threshold * 100);
  smartPositioning.checked = settings.smartPositioning;
  cssBackgrounds.checked = settings.cssBackgrounds;
  theme.value = settings.theme;
  renderSettings();
  renderReports(values.feedbackReports);
  rememberSections(values.openSections);
  chrome.runtime.sendMessage({ target: "background", type: "status" })
    .then(renderDetectorState)
    .catch(() => renderDetectorState({ enabled: values.enabled, warmingUp: false }));
  try {
    currentHost = new URL(tab.url).hostname;
    if (currentHost) {
      document.querySelector("#current-site").hidden = false;
      document.querySelector("#site-host").textContent = currentHost;
      const favicon = document.querySelector("#site-favicon");
      if (tab.favIconUrl) favicon.src = tab.favIconUrl;
      else favicon.hidden = true;
    }
  } catch {}
  renderExcludedSites();
});

document.querySelector("#settings-form").addEventListener("input", renderSettings);
document.querySelector("#settings-form").addEventListener("change", saveSettings);
filterSite.addEventListener("change", async () => {
  if (!currentHost) return;
  excludedSites = filterSite.checked
    ? excludedSites.filter((host) => host !== currentHost)
    : [...new Set([...excludedSites, currentHost])].sort();
  await chrome.storage.local.set({ excludedSites });
  renderExcludedSites();
});
document.querySelector("#back").addEventListener("click", () => { location.hash = ""; });
document.querySelector("#reports-info").addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  const tooltip = document.querySelector("#reports-tooltip");
  tooltip.hidden = !tooltip.hidden;
  event.currentTarget.setAttribute("aria-expanded", String(!tooltip.hidden));
});
addEventListener("hashchange", renderRoute);
chrome.runtime.onMessage.addListener((message) => {
  if (message?.target === "popup" && message.type === "state") renderDetectorState(message);
});
chrome.storage.onChanged.addListener((changes) => {
  if (changes.feedbackReports) renderReports(changes.feedbackReports.newValue ?? []);
  if (changes.excludedSites) {
    excludedSites = changes.excludedSites.newValue ?? [];
    renderExcludedSites();
  }
});
renderRoute();
