# Medical Image Triage

A web app that screens a frontal **chest X-ray** for a **pneumonia** pattern. You upload an image and get a positive or negative screening result. The neural network (a fine-tuned ResNet18) runs **entirely in your browser** through ONNX Runtime Web and WebAssembly, so the image never leaves your device.

> **Not a medical device.** This is an educational and portfolio demo of a research model. It is not clinically validated and must not be used for diagnosis or treatment decisions.

**Live demo:** _add your Render URL here after deploying, e.g. `https://medtriage.onrender.com`_

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

1. **Open the app.** Use the live URL, or run it locally (see below). It works on laptops and phones.
2. **Wait for "Model ready".** The first visit downloads the model (about 22 MB) and the WebAssembly runtime (about 3.5 MB compressed), with a progress bar. After that your browser caches them.
3. **Step 1 – choose what to screen.** Choose a category from the dropdown. For now there is one: *Chest X-ray — Pneumonia*.
4. **Step 2 – provide an image.** You can:
   - tap the upload area to choose a file (on a phone you can also take a photo), or
   - drag and drop an image onto it (desktop), or
   - click **Normal chest X-ray** or **Pneumonia chest X-ray** to try a sample from the held-out test set.

   JPEG, PNG, WebP and other browser-readable formats up to 40 MB are accepted.
5. **Step 3 – conversion.** A progress bar shows *"Converting image…"* while the image is decoded, converted to grayscale, resized to 256×256, center-cropped to 224×224 and normalized into a `[1, 3, 224, 224]` tensor. When it finishes, it shows *"✓ Conversion complete"* and a thumbnail of exactly what the model sees.
6. **Step 4 – result.** You get **POSITIVE** (pneumonia pattern detected) or **NEGATIVE**, plus:
   - the model score (0–1) on a bar with the 0.5 decision threshold marked,
   - the model's holdout sensitivity and specificity, for context,
   - the caveats.
7. To screen another image, upload a new one. The model stays loaded.

Developers can add `?test` to the URL (e.g. `https://…/?test`). This runs both sample images through the full pipeline and prints machine-readable `TEST-RESULT` JSON.

---

## Running locally

**With Docker** (same image as production):

```bash
docker build -t medtriage .
docker run --rm -p 10000:10000 medtriage
# open http://localhost:10000
```

**Without Docker:** any static file server works, e.g. `cd app && python3 -m http.server 8000`. It must be served over HTTP; opening `index.html` as a file won't work. A plain server doesn't send the cross-origin-isolation headers, so inference runs single-threaded, which is slightly slower.

---

## Repository layout

```
app/                    the deployed web app (static files only)
  index.html, app.js, styles.css
  registry.json         list of disease categories -> model file, preprocessing and threshold
  models/cxr_pneumonia.onnx   browser model (BatchNorm folded, fp16-stored weights)
  samples/              two holdout images for quick testing
  vendor/ort/           ONNX Runtime Web 1.30.0 (MIT)
src/                    data / training / evaluation / export pipeline (Python)
models/                 trained checkpoint (.pt), ONNX exports, training logs
reports/                holdout metrics, report and plots
data/splits.csv         patient-disjoint train/val/holdout assignment for every image
data/manifest.csv       splits.csv + path of each preprocessed image
deploy/nginx.conf.template  web server config used by the Docker image
Dockerfile, render.yaml     deployment
```

To add a new disease category later, add an ONNX model under `app/models/` and an entry in `app/registry.json`. The dropdown, preprocessing and result display are all driven by the registry.

---

## Getting the data

The images themselves (about 2.3 GB) are **not** in this repository. The split files `data/splits.csv` and `data/manifest.csv` are included, so the exact splits can be reproduced.

**Dataset:** Kermany, Zhang & Goldbaum (2018), *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, Mendeley Data, v2. DOI: [10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2). License: **CC BY 4.0**.

1. Download `ChestXRay2017.zip` from <https://data.mendeley.com/datasets/rscbjbr9sj/2>. The same images are mirrored on Kaggle as [`paultimothymooney/chest-xray-pneumonia`](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), but that mirror moves 16 images into an extra `val/` folder; if you use it, move them back into `train/` so paths match `data/splits.csv`.
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

## Deploying to Render

The `Dockerfile` builds a small `nginx:alpine` image that serves `app/`. Inference happens in the browser, so the server needs no Python and no GPU, and Render's free instance is enough. nginx listens on `$PORT`, which Render sets to 10000. It also sends the `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers, so the browser can use multi-threaded WebAssembly.

1. Sign in at <https://dashboard.render.com> with GitHub.
2. **New → Web Service**, then pick this repository. Alternatively, use **New → Blueprint**, which reads `render.yaml`.
3. Set **Language / Runtime** to **Docker**. Leave the Dockerfile path as `./Dockerfile` and the root directory empty.
4. Instance type: **Free**. No environment variables are needed.
5. **Deploy.** When the log shows the service is live, open the `https://<name>.onrender.com` URL.

On the free plan the service sleeps after 15 minutes without traffic. The first request after that takes roughly 30–60 seconds to wake it up.

---

## Credits and licenses

- Data: Kermany et al. 2018, CC BY 4.0. The sample images in `app/samples/` come from the same dataset.
- Runtime: [ONNX Runtime Web](https://github.com/microsoft/onnxruntime), MIT.
- Backbone: torchvision ResNet18 with ImageNet weights.
