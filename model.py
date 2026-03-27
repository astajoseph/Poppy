import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------
# 1. CLEAN VILLAGE NAMES
# ---------------------------
def clean_name(x):
    return str(x).strip().lower().replace(" ", "")

# ---------------------------
# 2. LOAD DATA
# ---------------------------
main_df = pd.read_csv("Main_Data.csv")
target_df = pd.read_csv("Targets_updated.csv")

main_df.columns = main_df.columns.str.strip()
target_df.columns = target_df.columns.str.strip()

main_df["Villl_name"] = main_df["Villl_name"].apply(clean_name)
target_df["Vill_name"] = target_df["Vill_name"].apply(clean_name)

# ---------------------------
# 3. MERGE ON VILLAGE
# ---------------------------
df = pd.merge(main_df, target_df, left_on="Villl_name", right_on="Vill_name", how="inner")
print("Columns in merged dataframe:\n", df.columns.tolist())
print(f"Merged dataset shape: {df.shape}")

df = df.drop(columns=["Villl_name", "Vill_name"], errors="ignore")

# ---------------------------
# 4. FEATURE & TARGET SELECTION
# ---------------------------
X = df.select_dtypes(include=np.number).drop(columns=["Kg_per_Hectare"], errors="ignore")
y = df["Kg_per_Hectare"]

print(f"\nFeatures used: {X.columns.tolist()}")
print(f"Target: Kg_per_Hectare | Samples: {len(y)}")

# ---------------------------
# 5. HANDLE MISSING DATA
# ---------------------------
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ---------------------------
# 6. SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ---------------------------
# 7. MODEL WITH DROPOUT + L2 + BATCHNORM
# ---------------------------
# Key anti-overfitting techniques applied:
# - Dropout after every hidden layer (rates increase deeper in the network)
# - L2 kernel regularization on Dense layers
# - BatchNormalization to stabilize activations
# - ReduceLROnPlateau to fine-tune learning near convergence
# - EarlyStopping with restore_best_weights

def build_model(input_dim):
    model = Sequential([
        # --- Block 1 ---
        Dense(256, activation="relu",
              kernel_regularizer=l2(1e-4),
              input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),          # Drop 40% of neurons

        # --- Block 2 ---
        Dense(128, activation="relu",
              kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.15),         # Drop 35% of neurons

        # --- Block 3 ---
        Dense(64, activation="relu",
              kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.1),          # Drop 30% of neurons

        # --- Output ---
        Dense(1)
    ])
    return model

model = build_model(X_train.shape[1])
model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

# ---------------------------
# 8. CALLBACKS
# ---------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=12,               # More patience to find real minimum
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,                # Halve LR when plateau detected
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# ---------------------------
# 9. TRAIN
# ---------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=400,
    batch_size=32,             # Larger batch = more stable gradients
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ---------------------------
# 10. EVALUATE
# ---------------------------
y_pred = model.predict(X_test).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n📊 FINAL RESULTS")
print(f"RMSE:     {rmse:.4f}")
print(f"MAE:      {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Check train vs val loss gap (overfitting diagnostic)
final_train_loss = history.history['loss'][-1]
final_val_loss   = history.history['val_loss'][-1]
gap = abs(final_val_loss - final_train_loss)
print(f"\nFinal Train Loss: {final_train_loss:.4f}")
print(f"Final Val Loss:   {final_val_loss:.4f}")
print(f"Train/Val Gap:    {gap:.4f}  {'✅ Low (minimal overfitting)' if gap < 500 else '⚠️ High (overfitting detected)'}")

# ---------------------------
# 11. LOSS PLOT
# ---------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'],     label='Train Loss',      linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss\n(Dropout + L2 + BatchNorm regularization)')
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()
print("Loss plot saved as loss_curve.png")

# ---------------------------
# 12. PREDICTED vs ACTUAL PLOT
# ---------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidths=0.4)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
plt.xlabel('Actual Kg/Hectare')
plt.ylabel('Predicted Kg/Hectare')
plt.title(f'Actual vs Predicted  (R² = {r2:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()
print("Scatter plot saved as actual_vs_predicted.png")

# ---------------------------
# 13. CORRELATION HEATMAP
# ---------------------------
feature_names = X.columns.tolist()
corr_df = pd.DataFrame(X_scaled, columns=feature_names)
corr_df["Kg_per_Hectare"] = y.values

plt.figure(figsize=(10, 8))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, square=True)
plt.title("Feature-Target Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()
print("Heatmap saved as correlation_heatmap.png")
