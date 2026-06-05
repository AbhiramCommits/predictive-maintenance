import numpy as np
import pandas as pd

np.random.seed(42)

n = 10_000
machine_ids = np.random.randint(1, 101, size=n)

temperature = np.random.normal(loc=75, scale=15, size=n)
temperature = np.clip(temperature, 30, 120)

vibration = np.random.exponential(scale=0.5, size=n)
vibration = np.clip(vibration, 0, 10)

pressure = np.random.normal(loc=100, scale=20, size=n)
pressure = np.clip(pressure, 50, 200)

rotational_speed = np.random.normal(loc=1500, scale=300, size=n)
rotational_speed = np.clip(rotational_speed, 500, 3000)

failure = np.zeros(n, dtype=int)
failure_idx = np.random.choice(n, size=int(n * 0.05), replace=False)
failure[failure_idx] = 1

df = pd.DataFrame({
    'machine_id': machine_ids,
    'temperature': temperature.round(2),
    'vibration': vibration.round(4),
    'pressure': pressure.round(2),
    'rotational_speed': rotational_speed.round(2),
    'failure': failure,
})

df.to_csv('data/sensor_telemetry.csv', index=False)

print(f"Shape: {df.shape}")
print(f"Class distribution:\n{df['failure'].value_counts()}")
print(f"Positive rate: {df['failure'].mean():.2%}")
