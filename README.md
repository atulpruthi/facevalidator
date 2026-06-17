---
title: Facevalidator
emoji: 📉
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: 5.45.0
app_file: app.py
pinned: false
license: mit
short_description: face validator
---

# FaceValidator

FaceValidator is a Python API for validating profile photos before they are accepted for profile use. The current modular API is built with Flask and runs a validation pipeline that checks for unsafe content, weapons, deepfakes, non-frontal poses, multiple people, and optionally celebrity matches.

The codebase also contains older or experimental entrypoints (`combined_validator_api.py`, `app.py`, `app_old.py`, and `app_facebook.py`). For API usage, start with `main.py` or `server.py`.

## Features

- JWT-protected validation endpoints.
- Complete image validation pipeline through `POST /validate`.
- Individual validation endpoints for NSFW/weapons, deepfake, pose, and celebrity checks.
- Hugging Face Transformers models for object detection and image classification.
- MediaPipe-based face and pose checks when MediaPipe is available.
- Optional fastai celebrity classifier loaded from `models/celebrity-classifier.pkl`.
- File upload validation with a 16 MB maximum upload size.
- Utility code for duplicate image hash tracking.

## Validation Pipeline

`POST /validate` runs these stages in order:

1. NSFW and weapons detection
2. Deepfake detection
3. Single-person, frontal-face, and frontal-pose validation
4. Celebrity detection, when the local classifier is available

If a stage fails, the API returns `valid: false`, a human-readable `reason`, the failing `stage`, and any available intermediate model results.

## Project Structure

```text
.
├── main.py                     # Flask API entrypoint that eagerly loads models
├── server.py                   # Flask API entrypoint with production-style startup logging
├── routes/api.py               # API routes and upload handling
├── config/settings.py          # Thresholds, upload settings, model paths, JWT settings
├── auth/manager.py             # JWT generation and validation
├── models/manager.py           # Model loading and shared model instances
├── validators/                 # NSFW, deepfake, pose, and celebrity validators
├── utils/helpers.py            # File, base64, image hash, and duplicate tracking helpers
├── combined_validator_api.py   # Older single-file API implementation
├── requirements.txt            # Python dependencies
└── start_api*.bat              # Windows startup scripts
```

## Requirements

- Python 3.10, as declared in `runtime.txt`.
- A virtual environment is recommended.
- Internet access on first startup so Transformers can download model weights from Hugging Face.
- Enough memory and disk space for PyTorch, Transformers, MediaPipe, and model caches.
- Optional local celebrity model at `models/celebrity-classifier.pkl`.

The application uses these remote models:

- `facebook/detr-resnet-50` for person detection.
- `Falconsai/nsfw_image_detection` for NSFW classification.
- `dima806/deepfake_vs_real_image_detection` for deepfake classification.
- `NabilaLM/detr-weapons-detection_40ep` for weapons detection.

## Setup

Create and activate a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set a JWT secret for local development:

```bash
export JWT_SECRET_KEY="replace-this-with-a-local-secret"
```

On Windows PowerShell:

```powershell
$env:JWT_SECRET_KEY = "replace-this-with-a-local-secret"
```

## Running the API

For the modular API with model loading during startup:

```bash
python main.py
```

For the production-style entrypoint:

```bash
python server.py
```

The API listens on:

```text
http://localhost:5000
```

## Authentication

Most validation endpoints require a JWT bearer token. For local testing, generate a token with:

```bash
curl -X POST http://localhost:5000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test_user","expires_in_hours":24}'
```

Use the returned token in the `Authorization` header:

```text
Authorization: Bearer <token>
```

## API Endpoints

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/` | No | API metadata and endpoint list |
| `GET` | `/health` | No | Model load status |
| `POST` | `/auth/token` | No | Generate a local test JWT |
| `POST` | `/validate` | Yes | Run the complete validation pipeline |
| `POST` | `/validate/nsfw` | Yes | Run NSFW and weapons checks only |
| `POST` | `/validate/deepfake` | Yes | Run deepfake detection only |
| `POST` | `/validate/pose` | Yes | Run single-person and frontal-pose validation only |
| `POST` | `/validate/celebrity` | Yes | Run celebrity detection only |

Image uploads should use multipart form data with either `file` or `image` as the field name.

## Example Requests

Run the full validation pipeline:

```bash
curl -X POST http://localhost:5000/validate \
  -H "Authorization: Bearer <token>" \
  -F "file=@/path/to/photo.jpg"
```

Run only pose validation:

```bash
curl -X POST http://localhost:5000/validate/pose \
  -H "Authorization: Bearer <token>" \
  -F "image=@/path/to/photo.jpg"
```

Check API health:

```bash
curl http://localhost:5000/health
```

## Configuration

Configuration lives in `config/settings.py`. Important values include:

| Setting | Default | Description |
| --- | --- | --- |
| `MAX_CONTENT_LENGTH` | `16 * 1024 * 1024` | Maximum upload size |
| `UPLOAD_FOLDER` | `temp_uploads` | Temporary upload directory |
| `ALLOWED_EXTENSIONS` | `png`, `jpg`, `jpeg`, `gif`, `bmp`, `webp` | Accepted file extensions |
| `JWT_SECRET_KEY` | Environment variable or development fallback | JWT signing secret |
| `NSFW_THRESHOLD` | `0.7` | NSFW classification threshold |
| `DEEPFAKE_THRESHOLD` | `0.7` | Deepfake classification threshold |
| `WEAPON_THRESHOLD` | `0.9` | Weapons detection threshold |
| `CELEBRITY_THRESHOLD` | `0.7` | Celebrity classifier confidence threshold |
| `PERSON_DETECTION_THRESHOLD` | `0.8` | Person detection confidence threshold |
| `MIN_PERSON_AREA` | `5000` | Minimum detected person area |
| `CELEBRITY_MODEL_PATH` | `./models/celebrity-classifier.pkl` | Optional fastai classifier path |

Duplicate tracking settings are also present in `Config` and `utils/helpers.py`, but the current route file does not expose duplicate management endpoints.

## Model Notes

Core validation requires the Hugging Face models to load successfully. The celebrity classifier is optional: if `models/celebrity-classifier.pkl` is missing or fails to load, celebrity detection returns a disabled result and the rest of the validators can still run.

MediaPipe is used for frontal face, pose, and person-mask processing. If MediaPipe cannot be imported, pose validation falls back to limited behavior in some checks.

## Development Notes

- `combined_validator_api.py` is the older monolithic implementation that the modular files were refactored from.
- `app.py` currently loads the celebrity classifier and predicts against a hard-coded local image path; it is not the main API entrypoint.
- `start_api.bat` and `start_api_mediapipe.bat` are Windows helper scripts with machine-specific paths that may need editing before use.
- There is no test suite in this repository yet. Add tests around route auth, upload validation, and each validator before making behavior-changing updates.
