import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay

df=pd.read_csv("data/cleaned_titanic.csv")
X = df[
    [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone",
        "Embarked_Q",
        "Embarked_S"
    ]
]

y = df["Survived"]
print(X.head())

print()

print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Features:", X_train.shape)
print("Testing Features :", X_test.shape)

print("Training Labels  :", y_train.shape)
print("Testing Labels   :", y_test.shape)

# Create the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)
joblib.dump(model, "models/logistic_model.pkl")

print("Model saved successfully!")

# Make predictions on the test data
predictions = model.predict(X_test)

print("Predictions:")
print(predictions[:10])
print("\nActual Values:")
print(y_test.iloc[:10].values)
accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy*100:.7f}%")

cm = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(figsize=(6,6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Did Not Survive", "Survived"],
    cmap="Blues",
    ax=ax
)

plt.title("Titanic Survival - Confusion Matrix")
plt.tight_layout()

plt.savefig("images/confusion_matrix.png", dpi=300)

plt.show()
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

coefficients = coefficients.sort_values(
    by="Coefficient",
    ascending=True
)

plt.figure(figsize=(8,6))

colors = [
    "green" if value > 0 else "red"
    for value in coefficients["Coefficient"]
]

plt.barh(
    coefficients["Feature"],
    coefficients["Coefficient"],
    color=colors
)

plt.axvline(0, color="black", linewidth=1)

plt.title("Feature Importance (Logistic Regression)")

plt.xlabel("Coefficient Value")

plt.tight_layout()

plt.savefig("images/feature_importance.png", dpi=300)

plt.show()

fig, ax = plt.subplots(figsize=(6,6))

RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test,
    ax=ax
)

plt.title("ROC Curve")

plt.tight_layout()

plt.savefig("images/roc_curve.png", dpi=300)

plt.show()

from sklearn.metrics import roc_auc_score

probabilities = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(
    y_test,
    probabilities
)

print(f"AUC Score : {auc:.3f}")

# Predicted survival probabilities
probabilities = model.predict_proba(X_test)[:, 1]

plt.figure(figsize=(8, 5))

plt.hist(probabilities, bins=20)

plt.title("Predicted Survival Probability Distribution")
plt.xlabel("Probability of Survival")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig("images/predicted_probability_distribution.png", dpi=300)

plt.close()