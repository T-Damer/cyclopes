const button = document.querySelector("#toggle");

function render(enabled) {
  button.textContent = `Filter: ${enabled ? "ON" : "OFF"}`;
  button.dataset.enabled = String(enabled);
}

chrome.storage.local.get({ enabled: true }).then(({ enabled }) => render(enabled));
button.addEventListener("click", async () => {
  const enabled = button.dataset.enabled !== "true";
  await chrome.storage.local.set({ enabled });
  render(enabled);
});
