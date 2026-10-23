
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load experiment results
results = pd.read_csv("results/comparison.csv")

os.makedirs("results", exist_ok=True)

# Print comparison table
print("\nModel Comparison:")
print(results.to_string(index=False))

# Plot test accuracy
plt.figure(figsize=(8, 5))
plt.bar(
    results["Model"],
    results["Test Accuracy (%)"],
    color=["skyblue", "orange"]
)

plt.title("CNN Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 100)

for i, value in enumerate(results["Test Accuracy (%)"]):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.savefig("results/accuracy_comparison.png", dpi=300)
plt.show()

print("\nGraph saved to results/accuracy_comparison.png")
