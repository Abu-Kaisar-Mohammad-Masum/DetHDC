import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from dethdc import DetHDC


def main():
    # Load Iris
    X, y = load_iris(return_X_y=True)
    X = X.astype(np.float32)

    # 5-fold stratified cross-validation repeated 10 times
    # Total evaluations = 5 x 10 = 50
    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42,
    )

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = DetHDC(
            dimensions=5000,
            epochs=5,
            lr=0.01,
            margin=0.2,
            seed=42,
            device="auto",
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        accuracies.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(
            f"Fold {fold_id:02d} | "
            f"Acc={acc*100:.2f}% | "
            f"Prec={precision*100:.2f}% | "
            f"Recall={recall*100:.2f}% | "
            f"F1={f1*100:.2f}%"
        )

    print("\n" + "=" * 52)
    print("DetHDC - Iris Repeated Stratified Cross-Validation")
    print("=" * 52)
    print("Dataset             : Iris")
    print("Folds               : 5")
    print("Repeats             : 10")
    print("Total evaluations   : 50")
    print("HV dimension        : 5000")
    print("Refinement epochs   : 5")
    print("Learning rate       : 0.01")
    print("Margin              : 0.2")
    print("Model seed          : 42")
    print("-" * 52)

    print(
        f"Accuracy : {np.mean(accuracies)*100:.2f} "
        f"± {np.std(accuracies)*100:.2f}%"
    )
    print(
        f"Precision: {np.mean(precisions)*100:.2f} "
        f"± {np.std(precisions)*100:.2f}%"
    )
    print(
        f"Recall   : {np.mean(recalls)*100:.2f} "
        f"± {np.std(recalls)*100:.2f}%"
    )
    print(
        f"Macro F1 : {np.mean(f1_scores)*100:.2f} "
        f"± {np.std(f1_scores)*100:.2f}%"
    )


if __name__ == "__main__":
    main()
