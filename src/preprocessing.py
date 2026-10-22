
import tensorflow as tf

def load_dataset():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.cifar10.load_data()
    )

    # Normalize pixel values to 0–1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Class names
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    return x_train, y_train, x_test, y_test, class_names


if __name__ == "__main__":
    x_train, y_train, x_test, y_test, class_names = load_dataset()

    print("Dataset loaded successfully!")
    print("Training images:", x_train.shape)
    print("Testing images:", x_test.shape)
    print("Number of classes:", len(class_names))
