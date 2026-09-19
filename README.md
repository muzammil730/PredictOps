# PredictOps — Predictive Maintenance for Industrial IoT

An end-to-end MLOps project that predicts the **Remaining Useful Life (RUL)** of industrial machines from sensor telemetry, using the NASA C-MAPSS (FD001) turbofan degradation dataset. The project covers the full lifecycle: model training, a FastAPI serving layer, containerization, cloud deployment on AWS SageMaker Serverless, infrastructure as code, CI/CD, and monitoring.

## Architecture

Sensor data (NASA C-MAPSS FD001)
-> Feature engineering (rolling stats)
-> RandomForestRegressor (RUL prediction)
-> FastAPI
   -> Dashboard (HTML/CSS/JS)
   -> Prometheus + Grafana
   -> AWS SageMaker Serverless (via Docker + ECR)

Infrastructure: Terraform
CI/CD: GitHub Actions (tests, Terraform plan/apply, scheduled retraining)


## Tech stack

- **ML**: pandas, scikit-learn (RandomForestRegressor), MLflow for experiment tracking
- **API**: FastAPI, Pydantic
- **Deployment**: Docker, Amazon ECR, Amazon SageMaker Serverless Inference
- **Infrastructure**: Terraform (ECR, IAM, SageMaker model/endpoint-config/endpoint)
- **CI/CD**: GitHub Actions (test suite, Terraform CI, scheduled model retraining)
- **Monitoring**: Prometheus (request metrics, latency, drift gauge) + Grafana dashboards
- **Frontend**: vanilla HTML/CSS/JS dashboard

## API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `GET /machines` | Fleet overview — health status per machine |
| `POST /predict-machine/{unit_id}` | RUL prediction for a specific machine |
| `POST /predict` | Raw feature-vector prediction |
| `GET /metrics` | Prometheus metrics |
| `GET /ping`, `POST /invocations` | SageMaker container contract |

## Model evaluation — methodology and honest results

This is deliberately documented in detail, because getting it wrong is a common and easy-to-miss mistake with time-series/lifecycle data.

**The mistake, caught and fixed:** the model was initially evaluated with a random row-level train/test split. Because each machine's cycles are highly correlated, this let near-identical rows from the same engine appear in both the training and test sets — a form of data leakage. That produced an inflated **R² = 0.97**.

**The fix:** evaluating on the NASA-provided *test* set — engines the model never saw during training at all — gives a realistic picture:

| | Random split (leaked) | Held-out engines (honest) |
|---|---|---|
| R² | 0.97 | 0.36 → **0.58** after RUL capping |
| RMSE | 8.7 cycles | 47.1 → **17.9** cycles |
| MAE | 5.6 cycles | 36.2 → **11.2** cycles |

**RUL capping:** early-life degradation isn't observable in the sensor readings, so training against raw (often very large) RUL values adds noise the model can't learn from. Capping the target at 125 cycles — a standard technique for this dataset — is a change to the training *target*, not the model's output, and is documented here for transparency. It improved held-out R² from 0.36 to 0.58, which is in line with published baseline results for simple regressors on this benchmark. Sequence-aware models (e.g. LSTM) typically do better; that's a natural next step, not implemented here.

**Dashboard data:** `/machines` and `/predict-machine` serve predictions against the NASA *test* split (engines truncated mid-life), not the training split (which runs every engine to failure and would trivially show near-zero RUL for everything).

## Known limitations

- Single-condition, single-fault-mode subset (FD001) — not validated on the other C-MAPSS subsets.
- RandomForest baseline; no sequence modeling of degradation trends.
- SageMaker Serverless: functionally correct, but cold starts add latency versus a provisioned endpoint (an intentional cost/latency trade-off for a portfolio project).

## Engineering issues hit during development

Kept here deliberately — these were real debugging sessions, not hypothetical:

- **ECR/SageMaker manifest mismatch**: modern `docker buildx` pushes OCI-format manifests by default; SageMaker only accepts Docker Distribution v2 Schema 2. Fixed with `--output ...,oci-mediatypes=false`.
- **Deprecated SageMaker instance types**: `ml.t2.medium` is deprecated for new endpoints, and several current-gen types have a default account quota of zero. Switched to **Serverless Inference**, which sidesteps both problems and better fits a low-traffic demo.
- **SageMaker container contract**: custom containers must respond to a `serve` entrypoint and expose `/ping` + `/invocations`; the initial container image only ran `uvicorn` directly.
- **CRLF line endings**: a `serve` shell script created on Windows had CRLF line endings, which Linux containers can't execute as a shebang script.
- **GitHub's 100 MB file limit**: the trained model (~248 MB) can't be committed directly; it's excluded via `.gitignore` and produced by the training pipeline instead.
- **CI missing derived data**: `features_train.csv` is a generated artifact (correctly `.gitignore`d), so the CI retraining workflow needs to run the preprocessing pipeline itself before training — it wasn't originally, causing `FileNotFoundError` in CI.
- **MLflow / scikit-learn serialization**: a newer MLflow default (`skops`-based serialization) flags RandomForest's internal C structures as untrusted, breaking model logging in CI. Fixed by explicitly requesting `serialization_format="pickle"`.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the training dataset from raw C-MAPSS data
python src/preprocess.py
python src/feature_engineering.py

# 3. Train the model
python src/model_training.py

# 4. Build the evaluation/demo dataset (mid-lifecycle "live" machines)
python src/preprocess_test.py
python src/feature_engineering_test.py

# 5. Run the API
uvicorn app.main:app --reload

# 6. Open frontend/index.html in a browser (with the API running on :8000)
```

### Deploying to AWS

```bash
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  --output type=image,name=<ECR_URI>:latest,push=true,oci-mediatypes=false .

cd terraform
terraform init
terraform apply
```

## Project structure

PredictOps/
  app/                  # FastAPI application
  src/                  # Data pipeline: preprocessing, feature engineering, training, drift detection
  frontend/             # Dashboard (HTML/CSS/JS)
  terraform/            # AWS infrastructure (ECR, IAM, SageMaker)
  data/                 # Raw C-MAPSS data (generated CSVs are gitignored)
  models/               # Trained model artifact (gitignored)
  tests/                # Test suite
  .github/workflows/    # CI, Terraform CI, scheduled retraining
  Dockerfile
  prometheus.yml