# DecodeLabs AI Internship - Project 2: Data Classification Using AI
# Pipeline: Load -> Explore -> Split -> Tune K (CV) -> Train -> Predict -> Evaluate -> Compare

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)


def main():
    print("==================================================")
    print("   PROJECT 2: DATA CLASSIFICATION USING AI (KNN)  ")
    print("==================================================\n")

    # ------------------------------------------------------------------
    # Step 1: Load and understand the dataset (Iris benchmark)
    # ------------------------------------------------------------------
    iris = load_iris()
    X, y = iris.data, iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    df["species"] = [iris.target_names[i] for i in y]

    print(f"[+] Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"[+] Target Classes: {list(iris.target_names)}")
    print(f"[+] Class counts  : {dict(zip(iris.target_names, np.bincount(y)))}")
    print(f"[+] Missing values: {int(df.isnull().sum().sum())}\n")

    print("--- Feature Summary Statistics ---")
    print(df.drop(columns="species").describe().round(2))
    print("\n--- First 5 Rows ---")
    print(df.head())
    print()

    # ------------------------------------------------------------------
    # Step 2: Split data (80% train, 20% test) - shuffled and stratified
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
    )
    print(f"[+] Train size: {len(X_train)} | Test size: {len(X_test)}\n")

    # ------------------------------------------------------------------
    # Step 3: Choose K using 5-fold cross-validation on TRAINING data only
    # The pipeline fits the scaler inside each fold, so there is no leakage.
    # ------------------------------------------------------------------
    k_values = range(1, 21)
    cv_scores = {}
    for k in k_values:
        pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        cv_scores[k] = cross_val_score(pipe, X_train, y_train, cv=5).mean()

    best_k = max(cv_scores, key=cv_scores.get)
    print(f"[+] Best K from cross-validation: {best_k} "
          f"(CV accuracy = {cv_scores[best_k]:.3f})\n")

    # Plot the "elbow" curve (error rate vs K)
    plt.figure(figsize=(7, 4))
    plt.plot(list(cv_scores.keys()), [1 - s for s in cv_scores.values()], marker="o")
    plt.axvline(best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
    plt.xlabel("K Value")
    plt.ylabel("CV Error Rate")
    plt.title("Choosing K: Error Rate vs K")
    plt.xticks(list(k_values))
    plt.legend()
    plt.tight_layout()
    plt.savefig("k_selection_curve.png", dpi=150)

    # ------------------------------------------------------------------
    # Step 4: Feature scaling (fit on train only) and train final model
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_scaled, y_train)
    print("[+] Model Training Completed.")

    # ------------------------------------------------------------------
    # Step 5: Predict on the unseen test set
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)

    # ------------------------------------------------------------------
    # Step 6: Evaluation
    # ------------------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n---> Test Accuracy : {accuracy * 100:.2f}%")
    print(f"---> Macro F1 Score: {macro_f1:.3f}\n")

    cm = confusion_matrix(y_test, y_pred)
    print("--- Confusion Matrix ---")
    print(cm)

    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Confusion matrix heatmap
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix (KNN, K={best_k})")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)

    # ------------------------------------------------------------------
    # Step 7: More reliable estimate - 5-fold CV on the full dataset
    # ------------------------------------------------------------------
    full_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k))
    full_cv = cross_val_score(full_pipe, X, y, cv=5)
    print(f"\n[+] 5-Fold CV on full data: mean = {full_cv.mean():.3f}, "
          f"std = {full_cv.std():.3f}")

    # ------------------------------------------------------------------
    # Step 8: Compare KNN with other algorithms (same CV protocol)
    # ------------------------------------------------------------------
    candidates = {
        f"KNN (K={best_k})": KNeighborsClassifier(n_neighbors=best_k),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }
    print("\n--- Algorithm Comparison (5-fold CV accuracy) ---")
    for name, clf in candidates.items():
        pipe = make_pipeline(StandardScaler(), clf)
        scores = cross_val_score(pipe, X, y, cv=5)
        print(f"{name:<22}: {scores.mean():.3f} (+/- {scores.std():.3f})")

    print("\n[+] Plots saved: k_selection_curve.png, confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()