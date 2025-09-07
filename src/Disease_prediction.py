#IMPORTING IMPORTANT MODULES
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
# %matplotlib inline

#READING THE DATASET
data = pd.read_csv('/Users/rameshkumar/Desktop/PROJECT_RESUME/Disease-prediction-and-Drug-recommendation/Data/Raw/Disease_Predict_Dataset.csv')
encoder = LabelEncoder()
data["disease"] = encoder.fit_transform(data["disease"])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
plt.figure(figsize=(18, 8))
sns.countplot(x=y)
plt.title("Disease Class Distribution Before Resampling")
plt.xticks(rotation=90)
plt.savefig("images/disease_class_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

#CROSS-VALIDATION WITH STRATIFIED K FOLD
if 'gender' in X_resampled.columns:
    le = LabelEncoder()
    X_resampled['gender'] = le.fit_transform(X_resampled['gender'])

X_resampled = X_resampled.fillna(0)

if len(y_resampled.shape) > 1:
    y_resampled = y_resampled.values.ravel()

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

cv_scoring = 'accuracy'  # you can also use 'f1_weighted', 'roc_auc_ovr' for multi-class
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    try:
        scores = cross_val_score(
            model,
            X_resampled,
            y_resampled,
            cv=stratified_kfold,
            scoring=cv_scoring,
            n_jobs=-1,
            error_score='raise' 
        )
        print("=" * 50)
        print(f"Model: {model_name}")
        print(f"Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean():.4f}")
    except Exception as e:
        print("=" * 50)
        print(f"Model: {model_name} failed with error:")
        print(e)


#Training Individual Models and Generating Confusion Matrices\
#SVM
svm_model = SVC()
svm_model.fit(X_resampled, y_resampled)
svm_preds = svm_model.predict(X_resampled)
cf_matrix_svm = confusion_matrix(y_resampled, svm_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix_svm, annot=True, fmt="d")
plt.title("Confusion Matrix for SVM Classifier")
plt.savefig("images/svm_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"SVM Accuracy: {accuracy_score(y_resampled, svm_preds) * 100:.2f}%")

#GAUS NAIVE BAYes
nb_model = GaussianNB()
nb_model.fit(X_resampled, y_resampled)
nb_preds = nb_model.predict(X_resampled)
cf_matrix_nb = confusion_matrix(y_resampled, nb_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix_nb, annot=True, fmt="d")
plt.title("Confusion Matrix for Naive Bayes Classifier")
plt.savefig("images/gnb_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Naive Bayes Accuracy: {accuracy_score(y_resampled, nb_preds) * 100:.2f}%")

#RANDOM FOREST
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)
rf_preds = rf_model.predict(X_resampled)
cf_matrix_rf = confusion_matrix(y_resampled, rf_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix_rf, annot=True, fmt="d")
plt.title("Confusion Matrix for Random Forest Classifier")
plt.savefig("images/rf_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Random Forest Accuracy: {accuracy_score(y_resampled, rf_preds) * 100:.2f}%")

#COMBINING PREDICTION 
from statistics import mode
final_preds = [mode([i, j, k]) for i, j, k in zip(svm_preds, nb_preds, rf_preds)]
cf_matrix_combined = confusion_matrix(y_resampled, final_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix_combined, annot=True, fmt="d")
plt.title("Confusion Matrix for Combined Model")
plt.savefig("images/combined_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Combined Model Accuracy: {accuracy_score(y_resampled, final_preds) * 100:.2f}%")

#CREATING PREDICTION FUNCTION
symptoms = X.columns.values
symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms)}

def predict_disease(input_symptoms):
    input_symptoms = input_symptoms.split(",")
    input_data = [0] * len(symptom_index)
    
    for symptom in input_symptoms:
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1

    input_df = pd.DataFrame([input_data], columns=symptoms)

    rf_pred = encoder.classes_[rf_model.predict(input_df)[0]]
    nb_pred = encoder.classes_[nb_model.predict(input_df)[0]]
    svm_pred = encoder.classes_[svm_model.predict(input_df)[0]]

    final_pred = mode([rf_pred, nb_pred, svm_pred])
    
    return {
        "Random Forest Prediction": rf_pred,
        "Naive Bayes Prediction": nb_pred,
        "SVM Prediction": svm_pred,
        "Final Prediction": final_pred
    }

# List of unique symptoms
symptoms_list = [
    "fever", "headache", "nausea", "vomiting", "fatigue",
    "joint_pain", "skin_rash", "cough", "weight_loss", "yellow_eyes"
]

# Create a dictionary with numbers as keys
symptom_dict = {str(i+1): symptom for i, symptom in enumerate(symptoms_list)}

if __name__ == "__main__":
    print("=== Disease Prediction System ===")
    print("Enter symptom numbers separated by commas (e.g., 1,2,7)")
    print("Type 'exit' anytime to quit.\n")

    # Show available symptoms once at the start
    for key, val in symptom_dict.items():
        print(f"{key}: {val}")

    while True:
        user_input = input("\nYour choice: ").strip()

        if user_input.lower() == "exit":
            print("👋 Exiting the system. Stay healthy!")
            break

        # Convert numbers to symptom names
        selected_symptoms = [
            symptom_dict[num.strip()]
            for num in user_input.split(",")
            if num.strip() in symptom_dict
        ]

        if selected_symptoms:
            symptom_string = ",".join(selected_symptoms)
            predictions = predict_disease(symptom_string)

            print("\n=== Predictions ===")
            for model, pred in predictions.items():
                print(f"{model}: {pred}")
        else:
            print("⚠️ Invalid input. Please enter valid numbers.")