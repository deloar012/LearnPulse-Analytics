from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer


NUMERIC_FEATURES = [
    "total_click",
    "active_days",
    "last_login_day",
    "avg_click_per_day",
    "days_since_last_login",
    "imd_band",
    "num_of_prev_attempts",
    "studied_credits",
    "date_registration",
    "date_unregistration",
    "late_registration",
    "active_duration",
    "score_mean",
    "score_min",
    "score_max",
    "score_std",
    "id_assessment_count",
    "is_banked_mean"
]

CATEGORICAL_FEATURES = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "age_band",
    "disability"
]


def create_preprocessor(
    numeric_strategy="median",
    categorical_strategy="most_frequent"
):

    # ---------------- Numeric ---------------- #

    if numeric_strategy == "knn":

        numeric_pipeline = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler())
        ])

    elif numeric_strategy == "constant":

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="constant",
                fill_value=-1
            )),
            ("scaler", StandardScaler())
        ])

    else:

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy=numeric_strategy
            )),
            ("scaler", StandardScaler())
        ])

    # ---------------- Categorical ---------------- #

    if categorical_strategy == "constant":

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="constant",
                fill_value="Missing"
            )),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

    else:

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="most_frequent"
            )),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

    # ---------------- Column Transformer ---------------- #

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),

            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )

        ],

        remainder="drop"

    )

    return preprocessor