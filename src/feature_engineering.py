import pandas as pd

df = pd.read_csv("data/sensor_telemetry.csv")

original_cols = set(df.columns)

for col in ["temperature", "vibration", "pressure"]:
    df[f"{col}_rolling_mean"] = (
        df.groupby("machine_id")[col].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    )
    df[f"{col}_rolling_std"] = (
        df.groupby("machine_id")[col].transform(lambda x: x.rolling(window=5, min_periods=1).std())
    )

vibration_threshold = df["vibration"].quantile(0.75)
df["high_vibration"] = (df["vibration"] > vibration_threshold).astype(int)

new_cols = sorted(set(df.columns) - original_cols)

df.to_csv("data/sensor_telemetry_engineered.csv", index=False)

print(f"Shape: {df.shape}")
print(f"New columns added: {new_cols}")
print(f"Vibration 75th percentile threshold: {vibration_threshold:.4f}")
print(f"High vibration flag distribution:\n{df['high_vibration'].value_counts()}")
