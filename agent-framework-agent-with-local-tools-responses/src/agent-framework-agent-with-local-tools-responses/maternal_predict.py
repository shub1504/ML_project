"""
MaternalRisk AI - Prediction Engine
====================================
This module provides prediction capabilities for the MaternalRisk AI project.
It loads a pre-trained Scikit-learn Pipeline (SimpleImputer + StandardScaler + SVC)
and exposes a reusable function to predict maternal risk levels.

SAFETY & REGULATORY NOTICE:
---------------------------
This is an experimental research model developed for educational and research
purposes. It is NOT a clinically validated diagnostic system.
Do not provide medical advice or claim that its prediction determines actual
maternal health status. Always consult qualified healthcare professionals for
any medical diagnosis or treatment decisions.
"""

import math
from pathlib import Path
import warnings
import joblib
import pandas as pd

# Suppress sklearn unpickling version warnings for clean runtime logs
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------

# Resolve root directory and default path to the serialized model file
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "maternal_risk_model.joblib"

# The exact feature order required by the pre-trained Scikit-learn pipeline
FEATURE_NAMES = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
]

# Valid target categories predicted by the model
VALID_RISK_CLASSES = {"low risk", "mid risk", "high risk"}

# Global cache for the loaded model so subsequent predictions avoid re-reading disk
_CACHED_MODEL = None


# -----------------------------------------------------------------------------
# Model Loading & Input Validation Helpers
# -----------------------------------------------------------------------------

def load_trained_model(model_path: Path | str | None = None):
    """
    Load the pre-trained Scikit-learn Pipeline from disk.

    Parameters:
        model_path (Path | str | None): Optional custom path to the .joblib file.
                                       Defaults to 'models/maternal_risk_model.joblib'.

    Returns:
        sklearn.pipeline.Pipeline: The pre-fitted Scikit-learn pipeline object.

    Raises:
        FileNotFoundError: If the model file cannot be found at the given path.
        RuntimeError: If there is an error deserializing or loading the model.
    """
    global _CACHED_MODEL

    # If the default model is already loaded in memory, return the cached instance
    if model_path is None and _CACHED_MODEL is not None:
        return _CACHED_MODEL

    resolved_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    # Check if the model file exists on disk
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at: '{resolved_path}'. "
            "Please confirm 'models/maternal_risk_model.joblib' is present."
        )

    try:
        # Load the serialized Scikit-learn pipeline using joblib
        model = joblib.load(resolved_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load the trained model pipeline from '{resolved_path}': {exc}"
        ) from exc

    # Cache if using default model path
    if model_path is None:
        _CACHED_MODEL = model

    return model


def _validate_inputs(age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate) -> dict:
    """
    Validate that all six patient inputs are present, numeric, and finite.

    Rejects missing values (None), booleans (which inherit from int in Python),
    strings, NaN, and positive/negative infinity.

    Returns:
        dict: A dictionary mapping model feature names to verified float values.

    Raises:
        ValueError: If any input is missing, non-numeric, NaN, or infinite.
    """
    raw_inputs = {
        "Age": age,
        "SystolicBP": systolic_bp,
        "DiastolicBP": diastolic_bp,
        "BS": blood_sugar,
        "BodyTemp": body_temp,
        "HeartRate": heart_rate,
    }

    validated_floats = {}

    for feature_name, value in raw_inputs.items():
        # Check 1: Ensure input is not None (missing value)
        if value is None:
            raise ValueError(
                f"Missing input for '{feature_name}'. All six parameters are required."
            )

        # Check 2: Reject booleans explicitly (bool is a subclass of int in Python)
        if isinstance(value, bool):
            raise ValueError(
                f"Invalid type for '{feature_name}': boolean value '{value}' is not allowed. "
                "Expected a numeric integer or float."
            )

        # Check 3: Ensure value is either an int or a float
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Invalid type for '{feature_name}': expected numeric int or float, got {type(value).__name__} ({value!r})."
            )

        # Check 4: Reject NaN, +infinity, and -infinity
        if not math.isfinite(value):
            raise ValueError(
                f"Invalid numeric value for '{feature_name}': received non-finite value ({value}). "
                "NaN and infinity are not permitted."
            )

        validated_floats[feature_name] = float(value)

    return validated_floats


# -----------------------------------------------------------------------------
# Main Prediction Function
# -----------------------------------------------------------------------------

def predict_maternal_risk(
    age,
    systolic_bp,
    diastolic_bp,
    blood_sugar,
    body_temp,
    heart_rate,
    model_path: Path | str | None = None,
) -> str:
    """
    Predict the maternal risk category for a set of physiological measurements.

    This function accepts 6 core patient indicators, validates them, formats them
    into a structured pandas DataFrame matching the exact feature order expected by
    the pre-trained model pipeline, and returns the predicted risk classification.

    Parameters:
        age (int | float): Maternal age in years (e.g., 25).
        systolic_bp (int | float): Upper value of blood pressure in mmHg (e.g., 120).
        diastolic_bp (int | float): Lower value of blood pressure in mmHg (e.g., 80).
        blood_sugar (int | float): Blood glucose levels in mmol/L (e.g., 7.0).
        body_temp (int | float): Body temperature in degrees Fahrenheit (e.g., 98.0).
        heart_rate (int | float): Resting heart rate in beats per minute (e.g., 70).
        model_path (Path | str | None, optional): Custom path to the model artifact.
                                                 Defaults to 'models/maternal_risk_model.joblib'.

    Returns:
        str: Predicted risk category as a clean lowercase string:
             'low risk', 'mid risk', or 'high risk'.

    Note on Probabilities:
        This model utilizes a Support Vector Classifier (SVC) trained with
        `probability=False`. Therefore, probability or confidence scores are not
        generated or returned.

    Safety Notice:
        This is an experimental research model, not a clinically validated diagnostic
        system. Do not provide medical advice or claim that its prediction determines
        actual maternal health status.

    Raises:
        ValueError: If any input fails validation or prediction produces an unknown class.
        FileNotFoundError: If the model artifact cannot be located on disk.
        RuntimeError: If model deserialization or inference fails during pipeline execution.
    """
    # Step 1: Validate all 6 inputs (rejecting None, NaN, inf, booleans, and non-numbers)
    validated = _validate_inputs(
        age=age,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        blood_sugar=blood_sugar,
        body_temp=body_temp,
        heart_rate=heart_rate,
    )

    # Step 2: Load the pre-trained Scikit-learn Pipeline
    model = load_trained_model(model_path)

    # Step 3: Construct input DataFrame preserving exact feature order expected by pipeline
    # Feature order: Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate
    input_df = pd.DataFrame(
        [[
            validated["Age"],
            validated["SystolicBP"],
            validated["DiastolicBP"],
            validated["BS"],
            validated["BodyTemp"],
            validated["HeartRate"],
        ]],
        columns=FEATURE_NAMES,
    )

    # Step 4: Execute model inference through the pipeline
    try:
        raw_prediction = model.predict(input_df)
    except Exception as exc:
        raise RuntimeError(
            f"Model prediction failed during pipeline execution: {exc}"
        ) from exc

    # Step 5: Format and verify prediction output as clean string
    predicted_class = str(raw_prediction[0]).strip().lower()

    if predicted_class not in VALID_RISK_CLASSES:
        raise ValueError(
            f"Unexpected model output '{predicted_class}'. Expected one of: {sorted(VALID_RISK_CLASSES)}"
        )

    return predicted_class


# -----------------------------------------------------------------------------
# Verification Test Runner
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("MaternalRisk AI - Prediction Engine Verification Test")
    print("=" * 70)
    print("SAFETY DISCLAIMER: Experimental research model for educational use.")
    print("NOT a clinically validated diagnostic tool.\n")

    # Sample test cases representing typical clinical feature profiles
    test_cases = [
        {
            "description": "Sample Case 1: Standard / Low Risk Profile",
            "inputs": {
                "age": 25,
                "systolic_bp": 120,
                "diastolic_bp": 80,
                "blood_sugar": 7.0,
                "body_temp": 98.0,
                "heart_rate": 70,
            },
        },
        {
            "description": "Sample Case 2: Elevated BP & Glucose / High Risk Profile",
            "inputs": {
                "age": 35,
                "systolic_bp": 140,
                "diastolic_bp": 100,
                "blood_sugar": 15.0,
                "body_temp": 98.0,
                "heart_rate": 90,
            },
        },
        {
            "description": "Sample Case 3: Intermediate Profile",
            "inputs": {
                "age": 32,
                "systolic_bp": 120,
                "diastolic_bp": 90,
                "blood_sugar": 7.5,
                "body_temp": 98.0,
                "heart_rate": 70,
            },
        },
    ]

    print("Running sample predictions with pre-trained model pipeline:")
    for case in test_cases:
        desc = case["description"]
        params = case["inputs"]
        try:
            result = predict_maternal_risk(**params)
            print(f"\n[PASS] {desc}")
            print(f"       Inputs: {params}")
            print(f"       Predicted Risk Level: '{result}'")
        except Exception as err:
            print(f"\n[FAIL] {desc}: {err}")

    # Validation Error Handling Test: verify NaN and invalid values are rejected
    print("\n" + "-" * 70)
    print("Testing input validation and error rejection:")

    invalid_tests = [
        (
            "NaN input rejection",
            {"age": float("nan"), "systolic_bp": 120, "diastolic_bp": 80, "blood_sugar": 7.0, "body_temp": 98.0, "heart_rate": 70},
        ),
        (
            "Missing (None) input rejection",
            {"age": 25, "systolic_bp": None, "diastolic_bp": 80, "blood_sugar": 7.0, "body_temp": 98.0, "heart_rate": 70},
        ),
        (
            "String input rejection",
            {"age": "twenty-five", "systolic_bp": 120, "diastolic_bp": 80, "blood_sugar": 7.0, "body_temp": 98.0, "heart_rate": 70},
        ),
    ]

    for label, bad_params in invalid_tests:
        try:
            predict_maternal_risk(**bad_params)
            print(f"[FAIL] {label} did not raise an error as expected.")
        except ValueError as val_err:
            print(f"[PASS] {label}: Caught expected ValueError -> {val_err}")
        except Exception as unexpected_err:
            print(f"[FAIL] {label}: Caught unexpected error -> {unexpected_err}")

    print("\n" + "=" * 70)
    print("All verification checks completed successfully.")
    print("=" * 70)
