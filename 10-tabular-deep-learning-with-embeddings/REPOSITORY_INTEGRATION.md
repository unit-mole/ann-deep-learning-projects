# Monorepo Integration

Place the project folder at the repository root:

```text
ann-deep-learning-projects/
└── 10-tabular-deep-learning-with-embeddings/
```

The GitHub Actions workflow must remain in the **repository-level** workflow folder:

```text
ann-deep-learning-projects/
└── .github/
    └── workflows/
        └── tabular-embedding-ci.yml
```

Do not keep a second copy of `tabular-embedding-ci.yml` inside the individual project folder. The workflow should reference the following project path wherever a project directory is required:

```text
10-tabular-deep-learning-with-embeddings
```

This includes workflow settings such as `paths`, `working-directory`, and `cache-dependency-path`.

For Streamlit Community Cloud, use this entrypoint:

```text
10-tabular-deep-learning-with-embeddings/app/streamlit_app.py
```

The deployment dependency file is intentionally stored beside the Streamlit entrypoint:

```text
10-tabular-deep-learning-with-embeddings/app/requirements.txt
```

Recommended repository layout:

```text
ann-deep-learning-projects/
├── .github/
│   └── workflows/
│       └── tabular-embedding-ci.yml
├── 09-multi-output-prediction-system/
└── 10-tabular-deep-learning-with-embeddings/
```

After replacing the corrected documentation files, the relevant Git commands are:

```bat
cd /d "C:\Users\atripathi\OneDrive - Veralto\Desktop\AI Codes\GIT Projects\ann-deep-learning-projects"
git add "10-tabular-deep-learning-with-embeddings" ".github\workflows\tabular-embedding-ci.yml"
git commit -m "Update tabular embeddings project documentation"
git pull --rebase origin main
git push origin main
```
