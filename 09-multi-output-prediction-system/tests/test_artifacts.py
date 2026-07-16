import json
import zipfile
from pathlib import Path
import joblib

ROOT=Path(__file__).resolve().parents[1]

def test_model_archive_and_preprocessor_dimensions():
    with zipfile.ZipFile(ROOT/"models"/"multi_output_model.keras") as archive:
        config=json.loads(archive.read("config.json"))
    input_layer=config["config"]["layers"][0]
    assert input_layer["config"]["batch_shape"][-1]==42
    preprocessor=joblib.load(ROOT/"models"/"preprocessor.joblib")
    assert len(preprocessor.get_feature_names_out())==42

def test_required_metadata_exists():
    metadata=json.loads((ROOT/"models"/"target_metadata.json").read_text())
    assert metadata["churn_threshold"]>0
    assert metadata["clv_std"]>0
    assert metadata["engagement_std"]>0
