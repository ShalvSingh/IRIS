# Iris Flower Prediction - Run & Health Check

Quick instructions to run the Flask app using the provided virtual environment and verify the model is loaded.

````markdown
# Iris Flower Prediction - Run & Health Check

[![CI](https://github.com/ShalvSingh/iris-flower-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/ShalvSingh/iris-flower-prediction/actions/workflows/ci.yml)

Quick instructions to run the Flask app using the provided virtual environment and verify the model is loaded.

Prerequisites
- Python 3.10+ (the project venv was created with this)
- A virtual environment at `.venv` (created already in this workspace)

Run the app using the workspace virtual environment (Windows PowerShell):

```powershell
# Activate the venv
.\.venv\Scripts\Activate.ps1

# (Optional) install dependencies if not present
pip install -r requirements.txt

# Run the app
python .\app\main.py
```

Health check endpoint
- Once the server is running, open or curl:

  http://127.0.0.1:5000/health

- Response:
  - 200 OK with JSON {"status":"ok","model_path":"..."} when model loaded
  - 503 when the model isn't loaded (useful for readiness probes)

Notes
- If `joblib.load` fails with ModuleNotFoundError: No module named 'sklearn', ensure `scikit-learn` is installed in the same Python environment used to run the app. Example:

```powershell
pip install scikit-learn==1.6.1 joblib
```

- Prefer running the app with `.venv` because that environment already contains scikit-learn and matched packages used to save the model.

Continuous Integration
- A GitHub Actions workflow (`.github/workflows/ci.yml`) runs the test suite (pytest) on push and pull requests to `main`.
- The workflow installs pinned dependencies from `requirements.txt` and runs `pytest -q`. Tests use Flask's test client so no external server is required.

If you want the CI status badge to show up (the one at the top of this README), push this branch to GitHub and open a PR or merge to `main` to trigger the workflow.

Test report artifacts
- The CI workflow uploads the JUnit XML test report as a workflow artifact named `junit-report` (path: `reports/junit.xml`).
- After a workflow run completes, open the Actions run for the job, expand the `Artifacts` section and download `junit-report` to inspect the `junit.xml` test output locally.


````
