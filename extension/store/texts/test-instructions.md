No account, API key, subscription, or external service is required.
Testing steps:
Install the extension and open its popup.
Enable Cyclopes globally and confirm that filtering is enabled for the current website.
Open a webpage containing images at least 256 px in source resolution.
Keep the tab focused and images visible in the viewport.
On first use, allow several seconds for the packaged ONNX model and WebAssembly runtime to initialize.
Eligible images will receive an “AI XX%” confidence badge. Images below the configured minimum size, videos, hidden images, and heavily occluded images are intentionally skipped.
Settings in the popup allow testers to change the confidence threshold, minimum image size, appearance, CSS background-image detection, and excluded websites.
To test local correction reports, right-click an image and select the Cyclopes AI or non-AI reporting action. Reports remain stored locally and are not uploaded.
All image preprocessing and inference run locally inside the browser. No image content, browsing history, or correction reports are transmitted to our servers. The extension uses an offscreen document because Manifest V3 service workers do not provide the canvas/DOM APIs required for local image preprocessing and ONNX inference.
AI detection is probabilistic. Compressed images, screenshots, memes, edited artwork, and unusual image formats may produce incorrect results. The confidence badge is informational and is not definitive proof of image origin.