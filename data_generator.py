import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 5000

# Generate synthetic system metrics
cpu_usage = np.random.uniform(5, 100, n_samples)
ram_usage = np.random.uniform(10, 100, n_samples)
disk_usage = np.random.uniform(5, 100, n_samples)
network_traffic = np.random.uniform(50, 1000, n_samples)
error_count = np.random.poisson(lam=3, size=n_samples)
temperature = np.random.uniform(25, 100, n_samples)
uptime_hours = np.random.uniform(1, 500, n_samples)

# Define failure logic (rule-based for label generation)
failure = (
    (cpu_usage > 85) |
    (ram_usage > 90) |
    (disk_usage > 90) |
    (error_count > 10) |
    (temperature > 80)
).astype(int)

# Create DataFrame
data = pd.DataFrame({
    "CPU_Usage": cpu_usage,
    "RAM_Usage": ram_usage,
    "Disk_Usage": disk_usage,
    "Network_Traffic": network_traffic,
    "Error_Count": error_count,
    "Temperature": temperature,
    "Uptime_Hours": uptime_hours,
    "Failure": failure
})

# Save to CSV
data.to_csv("system_metrics.csv", index=False)

print("system_metrics.csv generated successfully!")
print("Total Samples:", len(data))
print("Failure Count:", data["Failure"].sum())
print("Non-Failure Count:", (data["Failure"] == 0).sum())
