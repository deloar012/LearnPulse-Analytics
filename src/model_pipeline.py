from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from preprocessing_pipeline import create_preprocessor


import preprocessing_pipeline

print(preprocessing_pipeline.__file__)
print(preprocessing_pipeline.NUMERIC_FEATURES)


def create_model(
    model_name="gradient_boosting",
    numeric_strategy="median",
    categorical_strategy="most_frequent",
    random_state=42
):
    """
    Create a complete sklearn pipeline.

    Parameters
    ----------
    model_name : str
        logistic_regression
        decision_tree
        random_forest
        svm
        gradient_boosting

    numeric_strategy : str
        mean
        median
        most_frequent
        constant
        knn

    categorical_strategy : str
        most_frequent
        constant
    """

    # ---------------- Preprocessor ---------------- #

    preprocessor = create_preprocessor(
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy
    )

    # ---------------- Models ---------------- #

    models = {

        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),

        "decision_tree": DecisionTreeClassifier(
            random_state=random_state
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1
        ),

        "svm": SVC(
            probability=True,
            random_state=random_state
        ),

        "gradient_boosting": GradientBoostingClassifier(
            random_state=random_state
        )
    }

    if model_name not in models:
        raise ValueError(
            f"Model '{model_name}' is not supported."
        )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", models[model_name])
    ])

    return model