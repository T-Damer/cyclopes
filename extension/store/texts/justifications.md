Single purpose:
Cyclopes detects and labels AI-generated images on webpages entirely on-device, with optional browser-local correction reports and no image uploads.

activeTab:
Used when the user opens the popup to identify the active website and apply the user-controlled per-site filter toggle.

contextMenus:
Adds image context-menu actions for saving an AI or non-AI correction report locally on the device.

offscreen:
Runs the packaged ONNX Runtime Web and model in an offscreen document because Manifest V3 service workers lack the DOM and canvas environment required for local image preprocessing and inference.

scripting:
Injects the packaged content script into already-open pages after installation or update, avoiding a forced reload.

storage:
Stores settings, excluded websites, UI preferences, and optional user-created correction reports in browser-local storage. Reports are never uploaded.

Host permission:
Required to detect eligible images on websites the user visits and display local results. Processing is local and can be disabled globally or per website.

Remote code: No

Data6:Website content and Web history

Privacy URL:
https://github.com/T-Damer/cyclopes/blob/main/docs/PRIVACY.md