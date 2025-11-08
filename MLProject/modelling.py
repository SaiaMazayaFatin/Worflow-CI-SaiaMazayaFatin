import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking_uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="MLflow tracking URI (lokal atau DagsHub)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="CI - Predictive Maintenance Best XGBoost",
        help="Nama experiment di MLflow",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "predictive_maintenance_preprocessing"

    df = pd.read_csv(data_dir / "predictive_maintenance_preprocessing.csv")

    # Samakan dengan preprocessing sebelumnya: bersihkan nama kolom
    df.columns = (
        df.columns
        .str.replace(r"[\[\]<>]", "", regex=True)
        .str.replace(" ", "_")
    )

    X = df.drop("Target", axis=1)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    best_params = {
        "n_estimators": 441,
        "max_depth": 10,
        "colsample_bytree": 0.8513524070705439,
        "reg_lambda": 0.7723967032989834,
        "learning_rate": 0.014448810144561711,
        "min_child_weight": 1.343076681369564,
        "gamma": 1.2683155011683183,
        "reg_alpha": 0.49636614756168607,
        "subsample": 0.841927603767933,
    }
    scale_pos_weight = 28.52029520295203  # dari hasil tuning

    with mlflow.start_run(run_name="CI_Best_XGBoost") as run:
        run_id = run.info.run_id
        base_dir = Path(__file__).resolve().parent
        run_id_path = base_dir / "last_run_id.txt"

        with open(run_id_path, "w") as f:
            f.write(run_id)
        
        mlflow.log_artifact(run_id_path)

        model = XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight,
            **best_params,
        )

        # Training
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        if y_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_proba)
            except Exception:
                roc_auc = float("nan")
        else:
            roc_auc = float("nan")

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        mlflow.log_param("model_name", "XGBoost_Best_Params_CI")
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        for k, v in best_params.items():
            mlflow.log_param(k, v)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("tn", tn)
        mlflow.log_metric("fp", fp)
        mlflow.log_metric("fn", fn)
        mlflow.log_metric("tp", tp)

        # Simpan model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Simpan classification report sebagai artifact
        report_text = classification_report(y_test, y_pred, zero_division=0)
        report_path = base_dir / "classification_report_best_xgb_ci.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        mlflow.log_artifact(report_path)

        # Print ke console
        print("=== HASIL PELATIHAN CI (BEST XGBOOST) ===")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"ROC AUC  : {roc_auc:.4f}")
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


if __name__ == "__main__":
    main()
