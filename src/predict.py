import pandas as pd
import joblib

model = joblib.load("models/logistic_model.pkl")
print("=" * 50)
print("        TITANIC SURVIVAL PREDICTOR            ")
print("=" * 50)
pclass = int(input("Passenger Class (1/2/3): "))
sex = input("Sex (male/female): ").strip().lower()
age = float(input("Age: "))
sibsp = int(input("Number of Siblings/Spouses: "))
parch = int(input("Number of Parents/Children: "))
fare = float(input("Fare: "))
embarked = input("Embarked (C/Q/S): ").strip().upper()
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
sex = 1 if sex == "female" else 0
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0
new_passenger = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "FamilySize": family_size,
    "IsAlone": is_alone,
    "Embarked_Q": embarked_q,
    "Embarked_S": embarked_s
    
}])
prediction = model.predict(new_passenger)
probability = model.predict_proba(new_passenger)
print("\n" + "=" * 50)

if prediction[0] == 1:
    print("Prediction: PASSENGER IS LIKELY TO SURVIVE")
else:
    print("Prediction: PASSENGER IS LIKELY TO NOT SURVIVE")

print(f"Survival Probability: {probability[0][1] * 100:.2f}%")
print(f"Death Probability: {probability[0][0] * 100:.2f}%")

print("=" * 50)