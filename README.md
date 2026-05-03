# cGMP Cell Analysis Web Application

FastAPI + MongoDB workflow application for:
- image upload (single/batch)
- AI-based positive-cell segmentation/counting using the best trained model
- reviewer decisions (pass / reanalyze / reject)
- mandatory electronic signature for reviewer decisions
- immutable report checksum registry + verification API/UI
- audit trail and report generation (cGMP/21 CFR Part 11 oriented)
- role-based dashboard: user, reviewer, admin

## Project Structure

- `backend/app`: FastAPI backend
- `frontend/index.html`: responsive UI
- `backend/uploads`: raw uploads
- `backend/processed_images`: masks/heatmaps
- `backend/reports`: pass/reanalysis/rejection reports

## Quick Start

1) Install and start MongoDB locally.

2) Install backend dependencies:

```bash
cd d:/IPA/cgmp_app/backend
python -m pip install -r requirements.txt
```

3) Run backend:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

4) Open UI:
- open `d:/IPA/cgmp_app/frontend/index.html` in browser

## Seed Credentials

- username: `admin`
- password: `admin123`

## Workflow

1. User/Admin creates batch with batch code + details.
2. User uploads one/multiple photos (renamed with batch code + sequence).
3. User triggers analyze:
   - best model checkpoint loaded from `d:/IPA/runs_cv_full/production_model.pth`
   - segmentation mask + heatmap generated into `backend/processed_images/<batch_code>/`
   - cell counts persisted per image
4. User submits batch for review.
5. Reviewer decides:
   - must provide electronic signature (`username`, `password`, `meaning`, `reason`)
   - `pass`: save pass report in `backend/reports`
   - `reanalyze`: save reanalysis report and notify user
   - `reject`: save rejection report and notify user
6. Audit logs capture all lifecycle actions with hash-chaining.
7. Every saved report has:
   - immutable sidecar checksum (`.sha256`)
   - chained checksum record in MongoDB (`report_checksums`)
   - verification endpoint: `GET /api/reports/verify?report_path=...`

## Docker Deployment (FastAPI + Mongo + Nginx)

From `d:/IPA/cgmp_app`:

```bash
docker compose up -d --build
```

Open UI at:
- `http://localhost:8080`

Services:
- `nginx` (public): `http://localhost:8080`
- `backend` (internal): `backend:8000`
- `mongo` (internal): `mongo:27017`

Persistent data:
- `cgmp_app/backend_data/uploads`
- `cgmp_app/backend_data/processed_images`
- `cgmp_app/backend_data/reports`
- `mongo_data` volume

## Notes

- This is a strong production starter. For strict 21 CFR Part 11 validation, add:
  - formal dual-signature/second-review workflows where required by SOP
  - immutable WORM archival for audit/report files
  - full IQ/OQ/PQ validation package
  - policy-driven password/session controls, lockout, and MFA
  - PKI-backed digital signatures for reports
