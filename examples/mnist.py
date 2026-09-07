import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

from dethdc import DetHDC


def main():
    transform = transforms.ToTensor()

    mnist = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    X = mnist.data.numpy().astype(np.float32) / 255.0
    y = mnist.targets.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    model = DetHDC(
        dimensions=10000,
        epochs=5,
        lr=0.01,
        margin=0.2,
        refinement=True,
        seed=42,
        device="auto",
    )

    print("Training DetHDC...")
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    pred = model.predict(X_test)

    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, pred))


if __name__ == "__main__":
    main()
