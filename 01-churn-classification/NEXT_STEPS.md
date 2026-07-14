# Next Steps

## Phase 1: Run the Supplied Demo

1. Extract the project.
2. Open a terminal in `01-churn-classification`.
3. Run `scripts\setup_windows.bat`.
4. Run `scripts\run_app.bat`.
5. Test single and batch predictions.
6. Capture one clean screenshot of each tab.

## Phase 2: Retrain with the Improved Pipeline

1. Activate the environment.
2. Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

3. Run:

```bash
python -m src.train
```

4. Review the new `models/metadata.json`.
5. Compare recall, F1, PR-AUC, and the confusion matrix against the legacy model.
6. Restart Streamlit. The application automatically prefers `model.keras` and
   `preprocessor.joblib` when they exist.

## Phase 3: GitHub Preparation

1. Confirm the dataset source and license.
2. Replace placeholder links in `README.md`.
3. Add app screenshots under `assets/screenshots/`.
4. Add your LinkedIn and GitHub profile links.
5. Run `pytest -q`.
6. Create the GitHub repository `ann-deep-learning-projects`.
7. Push the full root folder.
8. Add repository topics:
   - artificial-neural-network
   - churn-prediction
   - deep-learning
   - tensorflow
   - keras
   - streamlit
   - binary-classification
   - customer-analytics

## Phase 4: Hosting

1. Deploy `01-churn-classification/app.py` on Streamlit Community Cloud.
2. Use Python 3.12.
3. Test the public URL in an incognito browser.
4. Add the live-demo URL to the README.
5. Add the project to your GitHub profile README.
