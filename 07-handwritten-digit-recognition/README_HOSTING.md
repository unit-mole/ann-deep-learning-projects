# Streamlit Community Cloud Deployment Guide

Streamlit Community Cloud is the recommended host for this project because the application is already written in Streamlit, the model is small enough to keep in Git, and deployment can be connected directly to the public GitHub repository.

## Deployment files

The deployment uses:

- `streamlit_app.py` — Community Cloud entrypoint.
- `app/streamlit_app.py` — full application code.
- `requirements.txt` — Python dependencies.
- `models/digit_recognition_model.keras` — trained model.
- `outputs/model_metrics.json` — metrics loaded dynamically by the app.
- `data/sample_digits/` — built-in examples.
- `.streamlit/config.toml` — theme and upload settings.

## Recommended runtime

Select **Python 3.12** in Streamlit Community Cloud advanced settings. The project pins TensorFlow 2.21.0 and Streamlit 1.59.2.

## Deploy from the monorepo

1. Push the complete `07-handwritten-digit-recognition` folder to:

   ```text
   unit-mole/ann-deep-learning-projects
   ```

2. Confirm that this model file is visible on GitHub:

   ```text
   07-handwritten-digit-recognition/models/digit_recognition_model.keras
   ```

3. Sign in to Streamlit Community Cloud with GitHub.
4. Select **Create app** and choose the repository and `main` branch.
5. Enter this app file path:

   ```text
   07-handwritten-digit-recognition/streamlit_app.py
   ```

6. Open **Advanced settings** and select Python 3.12.
7. Deploy the app.
8. Test both a built-in sample and a custom upload.
9. Copy the generated `https://<app-name>.streamlit.app` URL into the README, resume, LinkedIn, and portfolio.

## Local deployment check

From the project directory:

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Common deployment issues

### Model file not found

Verify the exact case-sensitive path and confirm that the `.keras` file was committed.

### Dependency build failure

Confirm that Community Cloud is using Python 3.12, then reboot the app after checking `requirements.txt`.

### App predicts uploaded images poorly

This is usually domain shift rather than a deployment error. Use a single centered digit with high contrast. The model was trained on clean MNIST images.

### Change Python after deployment

Community Cloud requires deleting and redeploying an app to change its Python version.

Official references:

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/get-started/tutorials/create-an-app
- https://www.tensorflow.org/install/pip
