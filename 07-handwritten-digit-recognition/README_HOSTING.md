# Streamlit Community Cloud Deployment Guide

Streamlit Community Cloud is the recommended host for this project because the application is written in Streamlit, the trained model is small enough to keep in Git, and deployment connects directly to the public GitHub repository.

**Live application:**  
https://ann-deep-learning-projects-gsnhfzexxframenzenm5rx.streamlit.app/

## Deployment Files

The deployment uses:

- `streamlit_app.py` — Streamlit Community Cloud entrypoint;
- `app/streamlit_app.py` — complete application implementation;
- `requirements.txt` — application dependencies;
- `models/digit_recognition_model.keras` — trained ANN model;
- `outputs/model_metrics.json` — model metrics loaded dynamically by the app;
- `data/sample_digits/` — built-in sample images;
- `.streamlit/config.toml` — theme and upload configuration.

## Recommended Runtime

Select **Python 3.13** in Streamlit Community Cloud advanced settings.

The locally validated and deployed environment uses:

```text
Python 3.13
TensorFlow 2.21.0
Streamlit 1.59.2
```

## Deploy from the Monorepo

1. Push the complete project folder to:

   ```text
   unit-mole/ann-deep-learning-projects
   ```

2. Confirm that the trained model is visible on GitHub:

   ```text
   07-handwritten-digit-recognition/models/digit_recognition_model.keras
   ```

3. Sign in to Streamlit Community Cloud using GitHub.
4. Select **Create app** and choose:

   ```text
   Repository: unit-mole/ann-deep-learning-projects
   Branch: main
   ```

5. Enter this exact main file path:

   ```text
   07-handwritten-digit-recognition/streamlit_app.py
   ```

   Do **not** select:

   ```text
   07-handwritten-digit-recognition/app/streamlit_app.py
   ```

   The root-level entrypoint is required so Streamlit Community Cloud can locate the project-level `requirements.txt`.

6. Open **Advanced settings** and select:

   ```text
   Python 3.13
   ```

7. Leave the Secrets field blank because this project does not require API keys or credentials.
8. Deploy the app.
9. Test a built-in sample, an uploaded image, the probability chart, and the CSV download.
10. Confirm the final public URL:

    ```text
    https://ann-deep-learning-projects-gsnhfzexxframenzenm5rx.streamlit.app/
    ```

## Local Deployment Check

Open Command Prompt inside:

```text
07-handwritten-digit-recognition
```

### Windows Command Prompt

The explicit virtual-environment interpreter avoids global Python and `PATH` conflicts:

```bat
python -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
```

The local application normally opens at:

```text
http://localhost:8501
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Deployment Validation

After the cloud application opens, verify:

1. the home page and model metrics load;
2. built-in Digit 7 is predicted correctly;
3. built-in Digit 9 is predicted correctly;
4. a user-uploaded digit is processed and scored;
5. original, grayscale, and 28×28 previews appear;
6. the probability chart and table load;
7. the probabilities CSV downloads successfully;
8. the app reloads successfully after a browser refresh.

## Common Deployment Issues

### `ModuleNotFoundError` for Plotly, TensorFlow, or another dependency

Confirm that the selected main file path is:

```text
07-handwritten-digit-recognition/streamlit_app.py
```

Selecting `app/streamlit_app.py` can prevent Community Cloud from detecting the project-level `requirements.txt`.

### Model file not found

Confirm that this case-sensitive path exists in GitHub:

```text
07-handwritten-digit-recognition/models/digit_recognition_model.keras
```

### Dependency installation failure

Confirm that:

- the application uses Python 3.13;
- `requirements.txt` is committed;
- the correct root-level Streamlit entrypoint is selected.

Then reboot the app from **Manage app**.

### TensorFlow CPU or GPU messages

Messages about oneDNN, CPU instruction optimization, or GPU support are informational. CPU inference is appropriate for this application.

### Uploaded images produce weak predictions

This is usually domain shift rather than a deployment error. Use one centered digit with high contrast and minimal background noise.

### Change the Python version after deployment

Streamlit Community Cloud requires deleting and redeploying the app to change its Python version.

## Updating the Live Application

After deployment, future updates follow the standard Git workflow:

```bat
git add <changed-files>
git commit -m "Describe the update"
git pull --rebase origin main
git push origin main
```

Streamlit Community Cloud detects changes pushed to the connected branch and refreshes the deployed application.

## Official References

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
- https://www.tensorflow.org/install/pip
