# Medical Image Triage

A web app that screens a frontal **chest X-ray** for a **pneumonia** pattern. You upload an image and get a positive or negative screening result. The neural network (a fine-tuned ResNet18) runs **entirely in your browser** through ONNX Runtime Web and WebAssembly, so the image never leaves your device.

It is a **standalone, offline-capable web app**. It is hosted for free on GitHub Pages as plain static files, and no server does any computation. After the first visit, a service worker keeps the whole app and model on your device, so the link keeps working with no internet connection. On a phone you can also add it to your home screen like a regular app.

**Live app:** https://yaqi998.github.io/medtriage/

> **Not a medical device.** This is an educational and portfolio demo of a research model. It is not clinically validated and must not be used for diagnosis or treatment decisions.

---

## Holdout performance

The holdout set has 873 images from 520 patients. No patient in it appears in the training or validation splits.

| Metric | threshold 0.5 (used by the app) |
|---|---|
| AUROC | 0.997 |
| Accuracy | 97.3% |
| Sensitivity | 99.1% |
| Specificity | 92.4% |

The full report, with confidence intervals, the confusion matrix and per-subtype results, is in [`reports/holdout_report.md`](reports/holdout_report.md). The plots are in `reports/`.

**Caveat:** the training data is paediatric X-rays (ages 1–5) from a single hospital. On adult X-rays, other body parts, other modalities or ordinary photos, the model's output is meaningless.

---

## How to use the app (step by step)

1. **Open the link** https://yaqi998.github.io/medtriage/ on a laptop or phone.
2. **Wait for "Model ready".** The first visit downloads the model (about 22 MB) and the WebAssembly runtime, with a progress bar. When the green line **"✓ Saved on this device. This page now works offline."** appears under the title, everything is stored locally. From then on the same link works without internet, e.g. in airplane mode.
3. *(Optional, phone)* **Install it.** On iPhone Safari, tap **Share → Add to Home Screen**. On Android Chrome, tap **⋮ → Install app** (or *Add to Home screen*). It then opens full-screen from its own icon.
4. **Step 1 – choose what to screen.** Choose a category from the dropdown. For now there is one: *Chest X-ray — Pneumonia*.
5. **Step 2 – provide an image.** You can:
   - tap the upload area to choose a file (on a phone you can also take a photo), or
   - drag and drop an image onto it (desktop), or
   - click **Normal chest X-ray** or **Pneumonia chest X-ray** to try a sample from the held-out test set.

   More labeled test images to download and upload are at https://yaqi998.github.io/medtriage/samples/ (10 random holdout X-rays, one per patient).

   JPEG, PNG, WebP and other browser-readable formats up to 40 MB are accepted.
6. **Step 3 – conversion.** A progress bar shows *"Converting image…"* while the image is decoded, converted to grayscale, resized to 256×256, center-cropped to 224×224 and normalized into a `[1, 3, 224, 224]` tensor. When it finishes, it shows *"✓ Conversion complete"* and a thumbnail of exactly what the model sees.
7. **Step 4 – result.** You get **POSITIVE** (pneumonia pattern detected) or **NEGATIVE**, plus:
   - the model score (0–1) on a bar with the 0.5 decision threshold marked,
   - the model's holdout sensitivity and specificity, for context,
   - the caveats.
8. To screen another image, upload a new one. The model stays loaded.

Developers can add `?test` to the URL (e.g. `https://…/medtriage/?test`). This runs both sample images through the full pipeline and prints machine-readable `TEST-RESULT` JSON.

---

## Deployment (GitHub Pages)

`.github/workflows/pages.yml` publishes the `app/` folder to GitHub Pages on every push to `main`. There are no servers to run or pay for.

One-time setup: in the repository go to **Settings → Pages → Build and deployment → Source** and choose **GitHub Actions**. After that, every `git push` redeploys within about a minute. Progress is shown under the **Actions** tab.

How offline mode and updates work:
- `app/sw.js` precaches every file the app needs. The workflow stamps it with the commit SHA, so each deployment gets a fresh cache name. Returning visitors fetch the new version the next time they open the link online, and the old cache is deleted.
- GitHub Pages can't set custom HTTP headers. The service worker adds the `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers itself, which enables multi-threaded WebAssembly from the second page load onwards. The first visit runs single-threaded, which is still fast.

## Running locally

The app is plain static files in `app/`. Any static file server works:

```bash
cd app && python3 -m http.server 8000
# open http://localhost:8000
```

It must be served over `http://localhost` or `https://`. Opening `index.html` as a file (`file://`) won't work, because browsers block the model download and the service worker there.

**With Docker** (optional, e.g. to self-host on your own server):

```bash
docker build -t medtriage .
docker run --rm -p 10000:10000 medtriage   # nginx listens on $PORT, default 10000
# open http://localhost:10000
```

---

## Repository layout

```
app/                    the web app (static files only; this folder is what gets published)
  index.html, app.js, styles.css
  sw.js                 service worker: offline cache + cross-origin isolation headers
  manifest.webmanifest, icons/   installable-app metadata
  registry.json         list of disease categories -> model file, preprocessing and threshold
  models/cxr_pneumonia.onnx   browser model (BatchNorm folded, fp16-stored weights)
  samples/              holdout images: 2 one-click samples + a gallery page (samples/index.html) of 10 to download
  vendor/ort/           ONNX Runtime Web 1.30.0 (MIT)
src/                    data / training / evaluation / export pipeline (Python)
models/                 trained checkpoint (.pt), ONNX exports, training logs
reports/                holdout metrics, report and plots
data/splits.csv         patient-disjoint train/val/holdout assignment for every image
data/manifest.csv       splits.csv + path of each preprocessed image
.github/workflows/pages.yml  publishes app/ to GitHub Pages on every push to main
Dockerfile, deploy/     optional nginx image for self-hosting
```

To add a new disease category later, add an ONNX model under `app/models/` and an entry in `app/registry.json`. Also add the model file to the `PRECACHE` list in `app/sw.js` so it is available offline. The dropdown, preprocessing and result display are all driven by the registry.

---

## Getting the data

The images themselves (about 2.3 GB) are **not** in this repository. The split files `data/splits.csv` and `data/manifest.csv` are included, so the exact splits can be reproduced.

**Dataset:** Kermany, Zhang & Goldbaum (2018), *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, Mendeley Data, v2. DOI: [10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2). License: **CC BY 4.0**.

1. Download `ChestXRay2017.zip` from <https://data.mendeley.com/datasets/rscbjbr9sj/2>. The same images are mirrored on Kaggle as [`paultimothymooney/chest-xray-pneumonia`](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), but that mirror moves 16 images into an extra `val/` folder. If you use it, move them back into `train/` so the paths match `data/splits.csv`.
2. Unzip it into `data/raw/` so you have:

   ```
   data/raw/chest_xray/train/{NORMAL,PNEUMONIA}/*.jpeg
   data/raw/chest_xray/test/{NORMAL,PNEUMONIA}/*.jpeg
   ```

## Reproducing the model

Python 3.10+; install the packages from `requirements.txt`. Run these from the repo root:

```bash
pip install -r requirements.txt
python src/make_split.py          # dedupe, group by patient, write data/splits.csv (70/15/15, seed 42)
python src/preprocess.py          # -> data/processed/*.png (256px grayscale) + data/manifest.csv
python src/train.py               # ResNet18 transfer learning -> models/pneumonia_resnet18.pt
python src/evaluate.py            # holdout metrics -> reports/
python src/plot_report.py         # ROC, confusion matrix, training curves
python src/export_onnx.py         # -> models/pneumonia_resnet18.onnx (checked against PyTorch)
python src/export_web_model.py    # -> models/pneumonia_resnet18_fp16w.onnx (copy to app/models/cxr_pneumonia.onnx)
python src/quantize_onnx.py       # optional int8 experiment
```

If you already have `data/splits.csv` and want the exact published split, skip `make_split.py`.

---

## Credits and licenses

- Data: Kermany et al. 2018, CC BY 4.0. The sample images in `app/samples/` come from the same dataset.
- Runtime: [ONNX Runtime Web](https://github.com/microsoft/onnxruntime), MIT.
- Backbone: torchvision ResNet18 with ImageNet weights.
