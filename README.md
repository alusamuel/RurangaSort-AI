# RurangaSort AI

RurangaSort AI is an end-to-end Machine Learning system that classifies waste images into categories such as cardboard, glass, metal, paper, plastic, and general waste.

The project demonstrates the full Machine Learning lifecycle:

- Data acquisition
- Data validation and preprocessing
- Exploratory data analysis
- Model training and comparison
- Model evaluation
- API development
- User-interface development
- Bulk data upload
- Model retraining
- Model versioning
- Production monitoring
- Docker containerization
- Cloud deployment
- Load testing with Locust

---

## Video Demo

YouTube link: `https://youtube.com/YOUR-VIDEO-ID` (replace after recording — see [Demonstration Video](#demonstration-video)).

## Live URLs

```text
Live application:   https://YOUR-APPLICATION-URL
API:                https://YOUR-API-URL
Swagger docs:       https://YOUR-API-URL/docs
GitHub repository:  https://github.com/USERNAME/rurangasort-ai
```

---

## Project Description

Incorrect waste sorting reduces recycling efficiency and increases environmental pollution. RurangaSort AI helps users identify the correct waste category by analyzing an uploaded image.

A user can upload one image and receive:

- The predicted waste category
- A confidence score
- The top three predictions
- The active model version
- The prediction response time

Authorized users can also upload multiple labelled images and trigger model retraining.

---

## Supported Waste Categories

The initial version classifies images into:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The number of categories can be expanded when more labelled data becomes available.

---

## Project Objectives

1. Build a Machine Learning classification model using image data.
2. Compare a baseline CNN with a transfer-learning model.
3. Evaluate the models using appropriate classification metrics.
4. Create a FastAPI service for predictions and monitoring.
5. Build a Streamlit user interface.
6. Allow users to upload one image for prediction.
7. Allow users to upload multiple labelled images for retraining.
8. Trigger retraining from the user interface.
9. Track model versions and evaluation results.
10. Deploy the application on a cloud platform.
11. Simulate high traffic using Locust.
12. Compare performance using one, two, and four API containers.

---

## Main Features

### Single-Image Prediction

- Predicted class
- Confidence score
- Top-three predictions
- Model version
- Prediction latency

### Dataset Visualizations

- Number of images per class
- Image brightness distribution
- Average RGB values
- Image dimensions and aspect ratios
- Training and validation performance
- Confusion matrix
- Prediction-confidence distribution
- API latency statistics

### Bulk Data Upload

The system validates ZIP structure, class folder names, image formats, corrupted images, duplicate files, file sizes, and unsafe paths.

### Model Retraining

An authorized user can press a button to trigger retraining. Retraining runs in a separate background worker so that prediction requests remain available.

### Model Monitoring

API status, model status, model uptime, active model version, last retraining time, total predictions, failed predictions, average latency, P95 latency, current training status.

### Production Evaluation

The active model can be evaluated using a fixed production test dataset. Results are stored and compared across model versions.

---

## Machine Learning Lifecycle

```text
Data Acquisition -> Data Validation -> Data Preprocessing -> Exploratory Data Analysis
   -> Model Training -> Model Evaluation -> Model Registration -> API Deployment
   -> Model Monitoring -> New Data Upload -> Model Retraining -> Candidate Evaluation
   -> Model Promotion or Rejection
```

---

## System Architecture

```text
                     +--------------------+
                     |   Streamlit UI     |
                     +---------+----------+
                               |
                               v
                     +--------------------+
                     |   Load Balancer    |
                     +---------+----------+
                               |
                 +-------------+-------------+
                 v             v             v
           +----------+  +----------+  +----------+
           | FastAPI  |  | FastAPI  |  | FastAPI  |
           | Instance |  | Instance |  | Instance |
           +----+-----+  +----+-----+  +----+-----+
                +-------------+-------------+
                              |
                +-------------+-------------+
                v                           v
         +-------------+           +---------------+
         |Active Model |           |  Redis Queue  |
         +-------------+           +-------+-------+
                                            |
                                            v
                                  +-------------------+
                                  |  Training Worker  |
                                  +---------+---------+
                                            |
                    +-----------------------+-----------------------+
                    v                       v                       v
              +-----------+           +-------------+        +-----------+
              | Dataset   |           |Model Store  |        |Metrics DB |
              +-----------+           +-------------+        +-----------+
```

The prediction API and training process are separated. This prevents model retraining from blocking user prediction requests.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Machine Learning | TensorFlow and Keras |
| Baseline model | Custom Convolutional Neural Network |
| Transfer-learning model | MobileNetV2 |
| Notebook | Jupyter Notebook |
| API | FastAPI |
| User interface | Streamlit |
| Background processing | Celery |
| Message broker | Redis |
| Database | SQLite (PostgreSQL-ready) |
| Model and dataset storage | Local storage or Amazon S3 |
| Containers | Docker and Docker Compose |
| Load balancing | Nginx |
| Cloud deployment | AWS ECS Fargate (or any Docker host) |
| Monitoring | Custom `/metrics` endpoint + CloudWatch |
| Load testing | Locust |
| Version control | Git and GitHub |

---

## Repository Structure

```text
mlpo/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── Dockerfile.worker
│
├── notebook/
│   └── rurangasort_training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── training.py
│   ├── evaluation.py
│   ├── prediction.py
│   ├── monitoring.py
│   └── model_registry.py
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   ├── dependencies.py
│   └── routes/
│       ├── __init__.py
│       ├── health.py
│       ├── prediction.py
│       ├── upload.py
│       ├── retraining.py
│       ├── evaluation.py
│       └── metrics.py
│
├── ui/
│   ├── app.py
│   └── pages/
│       ├── dashboard.py
│       ├── prediction.py
│       ├── visualizations.py
│       └── retraining.py
│
├── worker/
│   ├── __init__.py
│   ├── celery_app.py
│   └── tasks.py
│
├── data/
│   ├── raw/{cardboard,glass,metal,paper,plastic,trash}/
│   ├── processed/
│   ├── train/ validation/ test/
│   └── uploaded/
│
├── models/
│   ├── active/ candidate/ archived/
│   ├── class_names.json
│   └── model_metadata.json
│
├── reports/
│   ├── figures/ evaluation/ production_evaluation/ load_tests/
│
├── locust/
│   ├── locustfile.py
│   └── test_images/
│
├── tests/
│   └── test_*.py
│
├── scripts/
│   ├── generate_synthetic_dataset.py
│   └── download_trashnet.py
│
└── infrastructure/
    ├── nginx.conf
    ├── ecs-task-definition.json
    └── deployment-instructions.md
```

---

## Dataset

### Source

The reference dataset is **TrashNet** (Gary Thung & Mindy Yang, Stanford, CC BY 2.0), 6 classes: cardboard, glass, metal, paper, plastic, trash. The original ~3.5GB archive exceeded GitHub's git-lfs limit, so the upstream repo ([garythung/trashnet](https://github.com/garythung/trashnet)) now points at a Hugging Face mirror: [`garythung/trashnet`](https://huggingface.co/datasets/garythung/trashnet), which hosts the same `dataset-resized.zip` (42.8MB, 2,527 images) and `dataset-original.zip` (3.6GB) the original repo did.

- Download script: [scripts/download_trashnet.py](scripts/download_trashnet.py) — `python scripts/download_trashnet.py --huggingface` fetches the resized zip straight into `data/raw/` (add `--full-size` for the 3.6GB original). `--url`, `--kaggle`, and `--manual-dir` are also supported for alternate mirrors.
- Licence: CC BY 2.0 — attribute the original authors when redistributing.
- If disk space is tight, point `data/raw`, `data/train`, `data/validation`, `data/test`, and `data/processed` at a directory junction/symlink on a drive with more room — the code just follows whatever those paths resolve to.

### Synthetic fallback (for offline development / CI)

Because a fresh clone may not have network access or Kaggle credentials, [scripts/generate_synthetic_dataset.py](scripts/generate_synthetic_dataset.py) generates a small, deterministic, colour/texture-coded placeholder dataset with the *same directory structure*. This lets the entire pipeline (preprocessing → training → API → UI → retraining) run end-to-end without the real dataset. **Replace it with real images before reporting real metrics** — the notebook and README clearly separate "demo run" results from real ones.

### Raw dataset structure

```text
data/raw/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

### Bulk upload structure (retraining)

```text
new_training_data.zip
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

Rejected: unknown class folders, unsupported file types, corrupted images, empty ZIPs, duplicate images, unsafe ZIP paths (path traversal), oversized files.

---

## Data Preprocessing

1. Detect corrupted files.
2. Remove/report duplicate images (hash-based).
3. Convert images to RGB.
4. Resize images to `224 × 224`.
5. Normalize pixel values (`/255.0`).
6. Split into train/validation/test (70/15/15, stratified).
7. Augment only the training set (rotation, flip, zoom, shift, brightness).
8. Save the class-name mapping (`models/class_names.json`).
9. Save a dataset manifest (`data/processed/manifest.json`) for reproducibility.

---

## Dataset Visualizations & Feature Interpretation

Implemented in [src/preprocessing.py](src/preprocessing.py) (`compute_dataset_statistics`) and rendered in the notebook + Streamlit `Visualizations` page.

1. **Class distribution** — reveals class imbalance and possible model bias toward majority classes.
2. **Image brightness (mean grayscale intensity) per class** — reveals whether categories were photographed under different lighting, which a CNN can latch onto instead of shape.
3. **Average RGB channel means per class** — helps determine whether the model may be relying on colour (e.g. brown cardboard vs. clear glass) rather than object shape/texture.
4. **Image dimensions / aspect ratio distribution** — identifies inconsistencies from images sourced from different cameras, which affects resizing/cropping decisions.

---

## Model Development

Two models are trained and compared in the notebook and `src/model.py`:

### Baseline CNN

```text
Input(224,224,3) -> Conv2D -> MaxPool -> Conv2D -> MaxPool -> Conv2D -> MaxPool
   -> GlobalAveragePooling -> Dense(128, relu) -> Dropout -> Dense(num_classes, softmax)
```

### MobileNetV2 (transfer learning)

Two-stage training:

1. Freeze the MobileNetV2 (ImageNet) base, train a new classification head.
2. Unfreeze the top N layers, fine-tune end-to-end at a low learning rate.

The final model is selected from actual evaluation results (macro F1, not accuracy alone) — see [Model Results](#model-results).

---

## Model Evaluation Metrics

Computed in [src/evaluation.py](src/evaluation.py):

- Training / validation / test accuracy
- Precision, recall, F1 per class
- Macro precision / recall / F1
- Weighted F1
- Confusion matrix
- One-vs-rest ROC-AUC, PR-AUC
- Log loss
- Average & P95 prediction latency
- Model file size

Macro F1 is the primary model-selection metric because it weighs every class equally regardless of how many images it has.

## Model Results

Trained and evaluated end-to-end inside [notebook/rurangasort_training.ipynb](notebook/rurangasort_training.ipynb) on the real TrashNet dataset (2,524 images after dedup, 70/15/15 split: 1766/379/379 — see [Dataset](#dataset)), 15 epochs (baseline) / 8+5 epochs head+fine-tune (MobileNetV2), CPU only. The notebook is committed **with its outputs** — open it directly to see the plots and printed metrics below reproduced in full, including EDA charts, training curves, confusion matrices, and a visualised single-image prediction.

| Metric | Baseline CNN | MobileNetV2 |
|---|---:|---:|
| Test accuracy | 0.235 | 0.546 |
| Macro precision | 0.039 | 0.506 |
| Macro recall | 0.167 | 0.475 |
| Macro F1-score | **0.063** | **0.451** |
| Weighted F1-score | 0.089 | 0.510 |
| ROC-AUC (OvR macro) | 0.483 | 0.867 |
| PR-AUC (macro) | 0.188 | 0.583 |
| Log loss | 1.718 | 1.152 |
| Average latency | 84.7 ms | 89.5 ms |
| P95 latency | 92.5 ms | 98.6 ms |
| Model size | 1.3 MB | 20.7 MB |

**Selected model: MobileNetV2 (v1, currently active).** The baseline CNN, trained from scratch on ~1,766 real images, collapsed to almost always predicting the single majority class — an accuracy number that looks passable in isolation but a macro F1 of 0.063 that exposes the collapse immediately (0 recall on most classes), which is exactly why macro F1 (not accuracy) drives model selection here. MobileNetV2's ImageNet-pretrained features generalize far better from this small a dataset: macro F1 0.451, real (non-zero) recall on every class, at the cost of ~16x model size and marginally higher latency (both still well under the 500ms promotion threshold). Its weakest class is `trash` (only 21 test images, by far the smallest class) — see [Known Limitations](#known-limitations). Re-running the notebook will retrain both models from scratch with fresh random initialization, so exact numbers will vary run to run (we observed macro F1 for MobileNetV2 ranging ~0.45-0.54 across runs) — the qualitative conclusion (MobileNetV2 wins decisively, baseline collapses) reproduces reliably.

Full per-class precision/recall/F1 and the confusion matrix are in `models/active/metrics.json` after running the pipeline; regenerate this table anytime via the notebook, or with `python scripts/prepare_data.py && python -m src.training --model <name> --epochs ...` for each architecture.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness/readiness + uptime |
| `GET` | `/model-info` | Active model metadata |
| `POST` | `/predict` | Predict one uploaded image |
| `POST` | `/upload-data` | Upload bulk retraining ZIP |
| `POST` | `/retrain` | Trigger model retraining (background) |
| `GET` | `/training-status/{job_id}` | Poll retraining job status |
| `POST` | `/evaluate` | Evaluate the active model on the fixed test set |
| `GET` | `/metrics` | API + model monitoring metrics (Prometheus-style + JSON) |
| `GET` | `/dataset-summary` | Dataset statistics for visualizations |

Interactive docs: `http://localhost:8000/docs` (Swagger), `http://localhost:8000/redoc` (ReDoc).

### Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@locust/test_images/sample_plastic.jpg"
```

### Example response

```json
{
  "prediction": "plastic",
  "confidence": 0.9231,
  "top_predictions": [
    {"class": "plastic", "confidence": 0.9231},
    {"class": "glass", "confidence": 0.0512},
    {"class": "metal", "confidence": 0.0154}
  ],
  "model_version": "v1",
  "latency_ms": 42.8
}
```

---

## Retraining Pipeline

```text
Upload labelled ZIP -> Validate images -> Create new dataset version -> Trigger retraining
   -> Celery training task -> Train candidate model -> Evaluate candidate
   -> Compare with active model -> Promote or reject -> Update model registry
```

Retraining runs in a Celery worker process, never inside the API process, so predictions stay available during training.

## Model Promotion Rules ([src/model_registry.py](src/model_registry.py))

A candidate is promoted only when **all** hold:

- Macro F1 ≥ active model's macro F1 (configurable tolerance).
- No class's recall regresses beyond the accepted threshold.
- Prediction latency stays within the configured limit.
- The model file loads successfully and passes a smoke prediction.
- Evaluation ran on the fixed held-out test set (not training data).

If any check fails, the active model is left untouched and the candidate is archived with its evaluation report for inspection.

## Model Versioning

```text
models/
├── active/      # currently served
├── candidate/   # awaiting evaluation
└── archived/    # previous versions, kept for rollback/comparison
```

---

## Prerequisites

- Python 3.10+
- Git
- Docker & Docker Compose
- Redis (or run via Docker Compose)

## Setup

```bash
git clone https://github.com/USERNAME/rurangasort-ai.git
cd rurangasort-ai

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Windows PowerShell
Copy-Item .env.example .env
# Linux/macOS
cp .env.example .env
```

Never commit real passwords, tokens, or cloud credentials in `.env`.

## Get data (pick one)

```bash
# Real dataset (recommended, needs internet)
python scripts/download_trashnet.py

# OR fast offline placeholder data (deterministic synthetic images)
python scripts/generate_synthetic_dataset.py
```

## Preprocess + Train (script, mirrors the notebook)

```bash
python scripts/prepare_data.py            # validates data/raw, dedupes, splits 70/15/15
python -m src.training --model baseline_cnn --epochs 10
python -m src.training --model mobilenet_v2 --epochs 10
```

Both commands save into `models/candidate/`. Promote the winning one to `models/active/`
via the API (`POST /retrain` runs this automatically) or manually with:

```bash
python -c "from src.model_registry import promote_candidate; print(promote_candidate())"
```

Or open the notebook:

```bash
jupyter notebook notebook/rurangasort_training.ipynb
```

## Run the FastAPI service

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

`http://localhost:8000/docs`

## Run the Streamlit UI

```bash
streamlit run ui/app.py
```

`http://localhost:8501`

## Run Redis + Celery worker

```bash
docker run --name rurangasort-redis -p 6379:6379 -d redis:latest

# Linux/macOS
celery -A worker.celery_app worker --loglevel=info
# Windows
celery -A worker.celery_app worker --loglevel=info --pool=solo
```

## Tests

```bash
pytest
pytest --cov=src --cov=api --cov-report=term-missing
```

---

## Docker

```bash
docker compose up --build          # foreground
docker compose up --build -d       # background
docker compose ps
docker compose logs -f api
docker compose logs -f worker
docker compose down
```

## Scale API containers (horizontal scaling behind Nginx)

```bash
docker compose up --build --scale api=1
docker compose up --build --scale api=2
docker compose up --build --scale api=4
```

Nginx (`infrastructure/nginx.conf`, mounted into the `nginx` service in `docker-compose.yml`) load-balances across however many `api` replicas are running — running multiple containers without going through the `nginx` service does not demonstrate horizontal scaling, since Compose's internal DNS round-robins but Locust would only be hitting one resolved container per connection.

---

## Locust Load Testing

```bash
locust -f locust/locustfile.py --host http://localhost
```

Open `http://localhost:8089`, point at the Nginx host (`http://localhost`), not directly at a single API container.

### Scenarios

| API Containers | Concurrent Users | Spawn Rate |
|---:|---:|---:|
| 1 | 10 | 2/s |
| 1 | 50 | 5/s |
| 1 | 100 | 10/s |
| 2 | 10 | 2/s |
| 2 | 50 | 5/s |
| 2 | 100 | 10/s |
| 4 | 10 | 2/s |
| 4 | 50 | 5/s |
| 4 | 100 | 10/s |

Each scenario: same images, same duration, same spawn rate, repeated ≥3×, average the results, save the Locust CSV (`--csv=reports/load_tests/<containers>c_<users>u`), and note container CPU/memory (`docker stats`).

### Results (fill in after running)

| Containers | Users | Req/s | Median ms | P95 ms | P99 ms | Failure Rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | Pending | Pending | Pending | Pending | Pending |
| 1 | 50 | Pending | Pending | Pending | Pending | Pending |
| 1 | 100 | Pending | Pending | Pending | Pending | Pending |
| 2 | 10 | Pending | Pending | Pending | Pending | Pending |
| 2 | 50 | Pending | Pending | Pending | Pending | Pending |
| 2 | 100 | Pending | Pending | Pending | Pending | Pending |
| 4 | 10 | Pending | Pending | Pending | Pending | Pending |
| 4 | 50 | Pending | Pending | Pending | Pending | Pending |
| 4 | 100 | Pending | Pending | Pending | Pending | Pending |

Do not claim added containers improved throughput/latency until these numbers actually show it.

---

## Cloud Deployment

Recommended target: AWS ECS Fargate (`infrastructure/ecs-task-definition.json`, `infrastructure/deployment-instructions.md`). Any Docker host (Render, Railway, GCP Cloud Run, an EC2 box running `docker compose`) works with the same images.

| AWS Service | Purpose |
|---|---|
| Amazon ECR | Store Docker images |
| Amazon ECS Fargate | Run API, UI, worker containers |
| Application Load Balancer | Distribute requests across API tasks |
| Amazon S3 | Store uploaded data & model artifacts (stateless containers) |
| Amazon CloudWatch | Logs & metrics |
| AWS IAM | Least-privilege task roles |

The API is stateless — uploaded files and model versions must not depend solely on one container's local filesystem in production (mount S3 or an EFS volume; see `deployment-instructions.md`).

---

## Production Monitoring

`GET /metrics` + Streamlit `Dashboard` page expose: API health, uptime, active model version, last retraining time, retraining status, total/failed predictions, average & P95 latency, confidence distribution, class-prediction distribution, dataset size, newly uploaded image count.

## Production Evaluation

`POST /evaluate` loads the active model, runs it against the fixed production test set, computes the full metric suite, stores `reports/production_evaluation/<version>_metrics.json`, and appends to `reports/production_evaluation/model_comparison.csv` so versions are comparable over time.

---

## Security Considerations

- File-type / MIME validation on every upload
- Upload size limits (`MAX_UPLOAD_SIZE_MB`)
- Image corruption detection before use
- Safe ZIP extraction (blocks path traversal / zip-slip, absolute paths, symlinks)
- Pydantic request validation on all endpoints
- Secrets only via environment variables, never committed
- API-key auth on `/upload-data` and `/retrain`
- Structured request logging
- Model file integrity check before load (hash + shape check)
- Basic rate limiting on prediction endpoint

## Known Limitations

- Real dataset is modest (2,527 images, 137-594 per class) — `trash` has only 21 test images and the active model's recall on it is correspondingly weak (0.095-0.19 across runs); more data for that class specifically would likely help most
- Class imbalance across categories (see [Model Results](#model-results) and the class-distribution chart)
- Visual similarity between paper and cardboard, and between glass/metal/plastic (see confusion matrix in `models/active/metrics.json`)
- Background/lighting bias in source photos
- CPU-only inference latency in the default Docker image (~90ms/image for MobileNetV2 on this dev machine)
- A from-scratch CNN baseline needs real hyperparameter/architecture tuning to be competitive — as trained here it collapses to the majority class; that's why the transfer-learning model was selected, not treated as a tie-breaker
- Retraining compute cost
- Limited real production feedback loop until deployed with real traffic

## Future Improvements

- Collect more real-world waste images per class
- Additional waste categories
- User feedback / correction loop on wrong predictions
- Grad-CAM explanations in the UI
- Automated data/model drift detection
- Scheduled retraining (cron trigger in addition to manual button)
- Full model registry (MLflow) instead of the file-based registry
- Auth/RBAC across all write endpoints
- CI/CD pipeline
- GPU training support
- On-device/mobile inference export

---

## Demonstration Video

Should show: project intro → repo → dataset structure → preprocessing → visualizations → CNN training → MobileNetV2 training → evaluation/confusion matrix → single prediction → dashboard → bulk upload → retraining trigger → training status → candidate evaluation → promotion/rejection → API docs → uptime → Locust load test → 1 vs 2 vs 4 container comparison → cloud deployment → limitations.

YouTube link: `https://youtube.com/YOUR-VIDEO-ID`

---

## Submission Requirements

**First attempt:** ZIP of the full GitHub repository (source, notebook, README, requirements, Docker files, trained model file, Locust script, tests, evaluation reports, config examples — large raw datasets may be excluded, with download instructions provided instead).

**Second attempt:** GitHub repository URL — `https://github.com/USERNAME/rurangasort-ai`

## Final Submission Checklist

- [ ] GitHub repository created
- [ ] README completed with real results
- [ ] Dataset source & licence documented
- [ ] Notebook run end-to-end on real data
- [ ] Preprocessing explained
- [ ] ≥3 image features interpreted
- [ ] Baseline CNN trained
- [ ] MobileNetV2 trained
- [ ] Models compared with real metrics
- [ ] Confusion matrix included
- [ ] Prediction function implemented
- [ ] Model file saved
- [ ] FastAPI application created
- [ ] Streamlit interface created
- [ ] Model uptime displayed
- [ ] Bulk upload implemented
- [ ] Retraining button implemented
- [ ] Background retraining worker implemented
- [ ] Model versioning implemented
- [ ] Model promotion rules implemented
- [ ] Docker containers created
- [ ] Load balancer configured
- [ ] Locust flood testing completed (1/2/4 containers)
- [ ] Production evaluation demonstrated
- [ ] Cloud deployment completed
- [ ] Live application URL added
- [ ] YouTube demonstration uploaded
- [ ] Repository ZIP prepared
- [ ] GitHub URL prepared
- [ ] Secrets removed from the repository

---

## Author

**Name:** Samuel Rurangamirwa
**Student ID:** YOUR STUDENT ID
**Institution:** YOUR INSTITUTION
**Course:** Machine Learning
**Module:** Machine Learning Cycle
**Email:** YOUR EMAIL ADDRESS

## Licence

Developed for educational purposes. Add the correct source-code and dataset licences before publishing the repository publicly (TrashNet is CC BY 2.0 — attribute Gary Thung & Mindy Yang).

## Acknowledgements

Dataset creators (TrashNet — Gary Thung & Mindy Yang), TensorFlow/Keras, FastAPI, Streamlit, Docker, Locust, AWS, course instructors and reviewers.
