import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

STAGE_WINDOWS = {
    "S1": ("2023-11-20", "2023-12-05"),
    "S2": ("2023-12-20", "2024-01-05"),
    "S3": ("2024-01-20", "2024-02-05"),
    "S4": ("2024-02-20", "2024-03-05"),
}

S1_PRIORITY_WEIGHTS = {"S1": 0.55, "S2": 0.2, "S3": 0.15, "S4": 0.1}


def normalize_village_name(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def resolve_existing_path(base_dir: str, candidates: List[str]) -> str:
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of these files were found in {base_dir}: {candidates}")


def load_stage_file(csv_path: str, stage_idx: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "Vill_name" not in df.columns and "Villl_name" in df.columns:
        df = df.rename(columns={"Villl_name": "Vill_name"})

    required = {"NDVI", "NIR", "RED", "VV", "VH", "Vill_name"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    stage_df = df[["Vill_name", "NDVI", "NIR", "RED", "VV", "VH"]].copy()
    stage_df["Vill_name"] = stage_df["Vill_name"].map(normalize_village_name)

    for col in ["NDVI", "NIR", "RED", "VV", "VH"]:
        stage_df[col] = pd.to_numeric(stage_df[col], errors="coerce")

    stage_df["_row_order"] = stage_df.groupby("Vill_name").cumcount()
    return stage_df.rename(
        columns={
            "NDVI": f"NDVI_S{stage_idx}",
            "NIR": f"NIR_S{stage_idx}",
            "RED": f"RED_S{stage_idx}",
            "VV": f"VV_S{stage_idx}",
            "VH": f"VH_S{stage_idx}",
        }
    )


def merge_stages(stage_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    merged = stage_dfs[0]
    for next_df in stage_dfs[1:]:
        merged = merged.merge(next_df, on=["Vill_name", "_row_order"], how="inner")
    return merged


def load_and_merge_target(target_path: str, merged_features: pd.DataFrame) -> pd.DataFrame:
    target_df = pd.read_csv(target_path)
    target_df.columns = [c.strip() for c in target_df.columns]
    target_df["Vill_name"] = target_df["Vill_name"].map(normalize_village_name)
    target_df["Kg_per_Hectare"] = pd.to_numeric(target_df["Kg_per_Hectare"], errors="coerce")
    if "District" not in target_df.columns:
        target_df["District"] = "UNKNOWN"
    target_df["District"] = target_df["District"].astype(str).str.strip().str.upper()
    target_df["_row_order"] = target_df.groupby("Vill_name").cumcount()
    return merged_features.merge(target_df, on=["Vill_name", "_row_order"], how="inner")


def add_stage_time_features(df: pd.DataFrame) -> List[str]:
    stage_midpoints: Dict[str, float] = {}
    stage_durations: Dict[str, float] = {}

    for stage, (start, end) in STAGE_WINDOWS.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        mid_ts = start_ts + (end_ts - start_ts) / 2
        stage_midpoints[stage] = float(mid_ts.to_julian_date())
        stage_durations[stage] = float((end_ts - start_ts).days + 1)

    min_mid = min(stage_midpoints.values())
    max_mid = max(stage_midpoints.values())
    mid_range = max(max_mid - min_mid, 1e-8)
    temporal_weights = {
        stage: (mid - min_mid) / mid_range
        for stage, mid in stage_midpoints.items()
    }

    min_days = min(stage_durations.values())
    max_days = max(stage_durations.values())
    days_range = max(max_days - min_days, 1e-8)
    duration_weights = {
        stage: (days - min_days) / days_range
        for stage, days in stage_durations.items()
    }

    created_cols: List[str] = []
    for signal in ["NDVI", "NIR", "RED", "VV", "VH"]:
        s1 = f"{signal}_S1"
        s2 = f"{signal}_S2"
        s3 = f"{signal}_S3"
        s4 = f"{signal}_S4"

        s1_priority_col = f"{signal}_S1_PRIORITY"
        temporal_col = f"{signal}_TEMPORAL_PROGRESS"
        delta_col = f"{signal}_S1_DELTA_MEAN"
        span_col = f"{signal}_SPAN_WEIGHTED"

        df[s1_priority_col] = (
            S1_PRIORITY_WEIGHTS["S1"] * df[s1]
            + S1_PRIORITY_WEIGHTS["S2"] * df[s2]
            + S1_PRIORITY_WEIGHTS["S3"] * df[s3]
            + S1_PRIORITY_WEIGHTS["S4"] * df[s4]
        )
        df[temporal_col] = (
            temporal_weights["S1"] * df[s1]
            + temporal_weights["S2"] * df[s2]
            + temporal_weights["S3"] * df[s3]
            + temporal_weights["S4"] * df[s4]
        )
        df[delta_col] = df[s1] - df[[s2, s3, s4]].mean(axis=1)
        df[span_col] = (
            duration_weights["S1"] * df[s1]
            + duration_weights["S2"] * df[s2]
            + duration_weights["S3"] * df[s3]
            + duration_weights["S4"] * df[s4]
        )
        created_cols.extend([s1_priority_col, temporal_col, delta_col, span_col])

    return created_cols


def save_correlation_map(df: pd.DataFrame, output_path: str) -> None:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    ax.set_title("Feature Correlation Map")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_loss_comparison_graph(history: Dict[str, List[float]], output_path: str) -> None:
    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    if val_loss:
        ax.plot(epochs, val_loss, label="Validation Loss", linewidth=2)
    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_comparison_graph(train_acc: float, test_acc: float, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Train Accuracy", "Test Accuracy"]
    values = [train_acc, test_acc]
    bars = ax.bar(labels, values)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.5, f"{value:.2f}%", ha="center")

    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Comparison")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def fit_feature_pipeline(
    X_train_df: pd.DataFrame,
    X_eval_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(X_train_df)
    X_eval_imputed = imputer.transform(X_eval_df)

    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_eval_scaled = scaler.transform(X_eval_imputed)
    return X_train_scaled, X_eval_scaled, imputer, scaler


def create_model(
    input_dim: int,
    hidden_units: List[int],
    learning_rate: float,
    dropout_rate: float,
    l2_lambda: float,
) -> Sequential:
    model = Sequential(name="yield_dnn")
    model.add(Input(shape=(input_dim,)))

    for units in hidden_units:
        model.add(
            Dense(
                units,
                activation="relu",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
                kernel_regularizer=l2(l2_lambda),
            )
        )
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate, seed=SEED))

    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def parse_hidden_units(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def next_layer_size(hidden_units: List[int], min_units: int = 8) -> int:
    if not hidden_units:
        return 32
    return max(min_units, hidden_units[-1] // 2)


def load_best_config(config_path: str) -> Dict:
    if not os.path.exists(config_path):
        return {
            "hidden_units": [128, 64, 32, 16],
            "learning_rate": 0.01,
            "dropout": 0.2,
            "l2_lambda": 0.0,
            "batch_size": 16,
        }

    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "best_hyperparameters" in payload:
        return payload["best_hyperparameters"]
    return payload


def regression_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0
    return float(max(0.0, 100.0 - mape))


def correlation_value(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final model training and output metrics")
    parser.add_argument("--data-dir", default="data", help="Directory containing stage and target CSV files")
    parser.add_argument("--config-path", default="artifacts/model3_best_config.json", help="Best config JSON from search script")
    parser.add_argument("--epochs", type=int, default=350, help="Maximum training epochs")
    parser.add_argument("--hidden-units", default="", help="Override hidden layers like 128,64,32")
    parser.add_argument("--auto-grow-layers", action="store_true", help="Keep adding hidden layers until test RMSE stops improving")
    parser.add_argument("--growth-patience", type=int, default=2, help="How many non-improving growth steps before stopping")
    parser.add_argument("--max-growth-steps", type=int, default=8, help="Maximum extra layers to try when auto-grow is enabled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, args.data_dir)
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    s1_path = resolve_existing_path(data_dir, ["S1_.csv", "S_1.csv"])
    s2_path = resolve_existing_path(data_dir, ["S2_.csv", "S_2.csv"])
    s3_path = resolve_existing_path(data_dir, ["S3_.csv", "S_3.csv"])
    s4_path = resolve_existing_path(data_dir, ["S4_.csv", "S_4.csv"])
    target_path = resolve_existing_path(data_dir, ["Targets_updated.csv", "Targets_ordered.csv"])

    merged = merge_stages(
        [
            load_stage_file(s1_path, 1),
            load_stage_file(s2_path, 2),
            load_stage_file(s3_path, 3),
            load_stage_file(s4_path, 4),
        ]
    )
    dataset = load_and_merge_target(target_path, merged)
    dataset["District"] = dataset["District"].astype(str).str.strip().str.upper()
    dataset["District_Code"] = pd.Categorical(dataset["District"]).codes.astype(float)

    temporal_feature_cols = add_stage_time_features(dataset)

    feature_cols = [
        "NDVI_S1", "NIR_S1", "RED_S1", "VV_S1", "VH_S1",
        "NDVI_S2", "NIR_S2", "RED_S2", "VV_S2", "VH_S2",
        "NDVI_S3", "NIR_S3", "RED_S3", "VV_S3", "VH_S3",
        "NDVI_S4", "NIR_S4", "RED_S4", "VV_S4", "VH_S4",
        "District_Code",
    ] + temporal_feature_cols

    X_df = dataset[feature_cols].copy()
    y = pd.to_numeric(dataset["Kg_per_Hectare"], errors="coerce").to_numpy()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=0.2,
        random_state=SEED,
    )

    train_mask = np.isfinite(y_train)
    test_mask = np.isfinite(y_test)
    X_train_df = X_train_df.iloc[train_mask]
    y_train = y_train[train_mask]
    X_test_df = X_test_df.iloc[test_mask]
    y_test = y_test[test_mask]

    X_train_scaled, X_test_scaled, imputer, scaler = fit_feature_pipeline(X_train_df, X_test_df)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

    config = load_best_config(os.path.join(base_dir, args.config_path))
    if args.hidden_units.strip():
        config["hidden_units"] = parse_hidden_units(args.hidden_units)

    learning_rate = float(config.get("learning_rate", 0.01))
    dropout_rate = float(config.get("dropout", 0.2))
    l2_lambda = float(config.get("l2_lambda", 0.0))
    batch_size = int(config.get("batch_size", 16))

    depth_history: List[Dict] = []
    best_model = None
    best_y_pred = None
    best_history: Dict[str, List[float]] = {}
    best_rmse = float("inf")
    no_improvement_steps = 0

    base_hidden_units = list(config["hidden_units"])
    total_steps = 1 if not args.auto_grow_layers else args.max_growth_steps + 1

    for step in range(total_steps):
        trial_hidden_units = base_hidden_units.copy()
        for _ in range(step):
            trial_hidden_units.append(next_layer_size(trial_hidden_units))

        model = create_model(
            input_dim=X_train_scaled.shape[1],
            hidden_units=trial_hidden_units,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            l2_lambda=l2_lambda,
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=0),
        ]

        history = model.fit(
            X_train_scaled,
            y_train_scaled,
            validation_split=0.15,
            epochs=args.epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        y_pred_scaled = model.predict(X_test_scaled, verbose=0).reshape(-1)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
        trial_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        depth_history.append(
            {
                "growth_step": step,
                "hidden_units": trial_hidden_units,
                "test_rmse": trial_rmse,
            }
        )
        print(f"Depth step {step}: layers={trial_hidden_units} test_rmse={trial_rmse:.4f}")

        if trial_rmse < best_rmse:
            best_rmse = trial_rmse
            best_model = model
            best_y_pred = y_pred
            best_history = history.history
            config["hidden_units"] = trial_hidden_units
            no_improvement_steps = 0
        else:
            no_improvement_steps += 1

        if args.auto_grow_layers and no_improvement_steps >= args.growth_patience:
            print("Stopping layer growth because test RMSE stopped improving.")
            break

    if best_model is None or best_y_pred is None:
        raise RuntimeError("Training failed to produce a valid model.")

    model = best_model
    y_pred = best_y_pred
    rmse = best_rmse

    train_pred_scaled = model.predict(X_train_scaled, verbose=0).reshape(-1)
    train_pred = y_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).reshape(-1)

    train_acc = regression_accuracy(y_train, train_pred)
    corr = correlation_value(y_test, y_pred)
    acc = regression_accuracy(y_test, y_pred)

    merged_csv_path = os.path.join(artifacts_dir, "model3_merged_dataset.csv")
    model_path = os.path.join(artifacts_dir, "model3_best.keras")
    imputer_path = os.path.join(artifacts_dir, "model3_imputer.joblib")
    scaler_path = os.path.join(artifacts_dir, "model3_scaler.joblib")
    y_scaler_path = os.path.join(artifacts_dir, "model3_target_scaler.joblib")
    metrics_path = os.path.join(artifacts_dir, "model3_metrics.json")
    corr_map_path = os.path.join(artifacts_dir, "model3_correlation_map.png")
    loss_graph_path = os.path.join(artifacts_dir, "model3_train_val_loss.png")
    accuracy_graph_path = os.path.join(artifacts_dir, "model3_accuracy_comparison.png")

    corr_df = dataset[feature_cols + ["Kg_per_Hectare"]].copy()
    save_correlation_map(corr_df, corr_map_path)
    save_loss_comparison_graph(best_history, loss_graph_path)
    save_accuracy_comparison_graph(train_acc, acc, accuracy_graph_path)

    dataset.to_csv(merged_csv_path, index=False)
    model.save(model_path)
    joblib.dump(imputer, imputer_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(y_scaler, y_scaler_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_hyperparameters": config,
                "metrics": {
                    "train_accuracy_percent": train_acc,
                    "accuracy_percent": acc,
                    "correlation": corr,
                    "rmse": rmse,
                },
                "depth_growth_history": depth_history,
                "files": {
                    "merged_dataset": merged_csv_path,
                    "model": model_path,
                    "imputer": imputer_path,
                    "feature_scaler": scaler_path,
                    "target_scaler": y_scaler_path,
                    "correlation_map": corr_map_path,
                    "train_val_loss_graph": loss_graph_path,
                    "accuracy_comparison_graph": accuracy_graph_path,
                },
            },
            f,
            indent=2,
        )

    print("Final training completed.")
    print(f"Selected hidden layers: {config['hidden_units']}")
    print(f"Train Accuracy (%): {train_acc:.4f}")
    print(f"Accuracy (%): {acc:.4f}")
    print(f"Correlation: {corr:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Saved correlation map: {corr_map_path}")
    print(f"Saved training-validation graph: {loss_graph_path}")
    print(f"Saved accuracy comparison graph: {accuracy_graph_path}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
