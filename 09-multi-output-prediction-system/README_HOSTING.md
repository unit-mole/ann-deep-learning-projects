# Hosting Guide — Streamlit Community Cloud

## Recommendation

Use **Streamlit Community Cloud** for this project. The application is already written in Streamlit, the model is small enough for a public demo, and the platform connects directly to a GitHub repository.

## Files Required for Deployment

- `streamlit_app.py` — deployment entrypoint;
- `app/streamlit_app.py` — application implementation;
- `requirements.txt` — pinned Python dependencies;
- `models/multi_output_model.keras`;
- `models/preprocessor.joblib`;
- `models/target_metadata.json`;
- `data/sample_batch_input.csv`;
- `src/` modules.

## Recommended Python Version

Select **Python 3.12** in Streamlit Community Cloud Advanced settings. Use the same version locally.

## Deploy from the Monorepo

1. Push this folder to the main `ann-deep-learning-projects` repository.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Select **Create app**.
4. Choose the repository and `main` branch.
5. Set the entrypoint to:

```text
09-multi-output-prediction-system/streamlit_app.py
```

6. Open **Advanced settings** and select Python 3.12.
7. Deploy the app.
8. Test manual scoring, included batch scoring, CSV upload, and CSV download.
9. Add the final URL to this project README and the main repository README.

## Dependency Notes

The entrypoint is stored in the project folder so Streamlit can find the adjacent `requirements.txt`. Do not add another dependency file inside `app/` unless you intentionally want it to take precedence.

## Suggested Public URL

A typical shareable URL will resemble:

```text
https://multi-output-ann.streamlit.app
```

The actual URL is chosen or assigned during deployment.

## Troubleshooting

### Model cannot load

- Confirm Python 3.12 is selected.
- Confirm `keras`, `tensorflow`, and `h5py` installed successfully.
- Confirm Git LFS was not used incorrectly for the small `.keras` file.
- Confirm `models/multi_output_model.keras` exists in GitHub.

### Preprocessor warning

The project pins scikit-learn because serialized joblib artifacts should be loaded with the same library generation used to create them.

### App cannot find a file

Keep all paths repository-relative. The app uses `Path(__file__)` rather than the process working directory.

## Official References

- Streamlit deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
- App dependencies: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- File organization: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
