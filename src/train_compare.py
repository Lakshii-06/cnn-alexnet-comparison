
import os
import csv
import tensorflow as tf

from preprocessing import load_dataset
from simple_cnn import build_simple_cnn
from alexnet_inspired import build_alexnet


EPOCHS = 5
BATCH_SIZE = 64
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# Load the same dataset for both models
x_train, y_train, x_test, y_test, class_names = load_dataset()

os.makedirs("results", exist_ok=True)

models_to_train = {
    "Simple CNN": build_simple_cnn(),
    "AlexNet-inspired CNN": build_alexnet()
}

results = []

for model_name, model in models_to_train.items():
    print(f"\nTraining: {model_name}")
    print("Parameters:", model.count_params())

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(
        x_test, y_test, verbose=1
    )

    results.append({
        "Model": model_name,
        "Parameters": model.count_params(),
        "Test Accuracy (%)": round(test_accuracy * 100, 2),
        "Test Loss": round(test_loss, 4),
        "Final Validation Accuracy (%)": round(
            history.history["val_accuracy"][-1] * 100, 2
        )
    })

    model.save(
        f"results/{model_name.lower().replace(' ', '_')}.keras"
    )

# Save comparison results
with open("results/comparison.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nFINAL MODEL COMPARISON")
for result in results:
    print(result)

print("\nResults saved in the results folder.")
