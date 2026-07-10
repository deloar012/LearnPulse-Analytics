import joblib
import pandas as pd


def load_model(model_path="models/decision_tree_pipeline.pkl"):
    """
    Load a trained sklearn pipeline.
    """
    return joblib.load(model_path)


def predict(model, data):
    """
    Predict final_result for new data.

    Parameters
    ----------
    model : trained sklearn pipeline

    data : pandas.DataFrame

    Returns
    -------
    predictions
    """

    predictions = model.predict(data)

    return predictions


if __name__ == "__main__":

    # ===========================
    # Load Model
    # ===========================

    model = load_model()

    # ===========================
    # Example Data
    # ===========================

    sample = pd.DataFrame({

        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "total_click": [850],
        "active_days": [110],
        "last_login_day": [260],
        "avg_click_per_day": [7.7],
        "days_since_last_login": [5],
        "gender": ["M"],
        "region": ["East Anglian Region"],
        "highest_education": ["A Level or Equivalent"],
        "imd_band": [40],
        "age_band": ["0-35"],
        "num_of_prev_attempts": [0],
        "studied_credits": [60],
        "disability": ["N"],
        "date_registration": [-15],
        
        "late_registration": [0],
        "active_duration": [260],
        "withdrew": [0],
        "score_mean": [75],
        "score_min": [60],
        "score_max": [95],
        "score_std": [8],
        "id_assessment_count": [6],
        "is_banked_mean": [0]

    })

    # ===========================
    # Prediction
    # ===========================

    prediction = predict(model, sample)

    print("=" * 60)
    print("Prediction")
    print("=" * 60)
    print(prediction)