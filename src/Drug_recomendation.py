# !pip install pandas numpy scikit-learn joblib tqdm matplotlib
#Import necessary library
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =====================================
# STEP 1: Load datasets & Create Mapping (Category-based)
# =====================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load base datasets
diseases = pd.read_csv("/Users/rameshkumar/Desktop/PROJECT_RESUME/Disease-prediction-and-Drug-recommendation/Data/Raw/Disease_Dataset_Category.csv")
drugs = pd.read_csv("/Users/rameshkumar/Desktop/PROJECT_RESUME/Disease-prediction-and-Drug-recommendation/Data/Raw/Drug_Dataset.csv")

# Clean categories to avoid mismatch
diseases["Category"] = diseases["Category"].str.strip().str.lower()
drugs["Category"] = drugs["Category"].str.strip().str.lower()

# Parameters
min_drugs = 2
max_drugs = 3

rows = []
for _, disease in diseases.iterrows():
    disease_id = disease["Disease_ID"]
    disease_cat = disease["Category"]

    # Filter drugs belonging to the same category
    possible_drugs = drugs[drugs["Category"] == disease_cat]

    # if possible_drugs.empty:
    #     print(f" No drugs found for DiseaseID {disease_id} with category {disease_cat}")
    #     continue  # skip if no matching drugs

    # Select 2–3 drugs from the matching category
    n_drugs = min(len(possible_drugs), np.random.randint(min_drugs, max_drugs + 1))
    chosen_drugs = np.random.choice(possible_drugs["Drug_ID"], size=n_drugs, replace=False)

    # Assign random percentages that sum to 100
    percentages = np.random.dirichlet(np.ones(n_drugs), size=1)[0] * 100
    percentages = np.round(percentages, 2)

    for drug_id, perc in zip(chosen_drugs, percentages):
        rows.append({
            "DiseaseID": disease_id,
            "DrugID": drug_id,
            "Percentage": perc
        })

# Build mapping dataframe
mapping_df = pd.DataFrame(rows)

if mapping_df.empty:
    raise ValueError(" Mapping failed: No rows generated. Check category alignment!")

# Normalize percentages per disease
mapping_df["Percentage"] = mapping_df.groupby("DiseaseID")["Percentage"].transform(
    lambda x: (x / x.sum() * 100).round(2)
)

# Save for reuse
mapping_df.to_csv("/Users/rameshkumar/Desktop/PROJECT_RESUME/Disease-prediction-and-Drug-recommendation/Data/Processed/disease_drug_map.csv", index=False)
print(" disease_drug_map.csv created successfully (Category-based mapping)!")

# =====================================
# STEP 2A: Prepare Training Data
# =====================================
merged_df = pd.merge(mapping_df, diseases, left_on="DiseaseID", right_on="Disease_ID", how="left")

# Features (X)
X = pd.get_dummies(merged_df.drop(["DiseaseID", "DrugID", "Percentage", "Disease_ID"], axis=1))

# Targets (y)
y_pivot = merged_df.pivot(index="DiseaseID", columns="DrugID", values="Percentage").fillna(0)

# Align features
X = X.groupby(merged_df["DiseaseID"]).mean()
X = X.reindex(y_pivot.index).fillna(0)

y = y_pivot.values

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# -----------------------------
# STEP 2B: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# STEP 3A: Random Forest Model
# -----------------------------
rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)

print("\n Random Forest Performance:")
print(f"R² Score: {rf_r2:.4f}")
print(f"RMSE: {rf_rmse:.4f}")
print(f"MAE: {rf_mae:.4f}")

# -----------------------------
# STEP 3B: SVM Model
# -----------------------------
svm_model = MultiOutputRegressor(SVR(kernel="rbf", C=10, gamma="scale"))
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)
svm_r2 = r2_score(y_test, svm_pred)
svm_rmse = np.sqrt(mean_squared_error(y_test, svm_pred))
svm_mae = mean_absolute_error(y_test, svm_pred)

print("\n SVM Performance:")
print(f"R² Score: {svm_r2:.4f}")
print(f"RMSE: {svm_rmse:.4f}")
print(f"MAE: {svm_mae:.4f}")


# -----------------------------
# STEP 4: Comparison Chart
# -----------------------------
metrics = ["R² Score", "RMSE", "MAE"]
rf_values = [rf_r2, rf_rmse, rf_mae]
svm_values = [svm_r2, svm_rmse, svm_mae]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8,6))
plt.bar(x - width/2, rf_values, width, label="Random Forest", color="forestgreen")
plt.bar(x + width/2, svm_values, width, label="SVM", color="royalblue")

plt.xticks(x, metrics)
plt.ylabel("Score Value")
plt.title("Random Forest vs SVM Performance Comparison")
plt.legend()
plt.savefig("images/disease_class_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------
# STEP 5: Save Models
# -----------------------------
joblib.dump(rf_model, "/Users/rameshkumar/Desktop/PROJECT_RESUME/Disease-prediction-and-Drug-recommendation/Data/Processed/Processedrf_model.pkl")
joblib.dump(svm_model, "/Users/rameshkumar/Desktop/PROJECT_RESUME/Disease-prediction-and-Drug-recommendation/Data/Processed/Processedsvm_model.pkl")


# -----------------------------
# STEP 6: Prediction Function
# -----------------------------
def predict_drugs_by_name(disease_name: str, model_choice="rf"):
    # Select model
    model = rf_model if model_choice.lower() == "rf" else svm_model

    # Get disease row
    disease_row = diseases[diseases["Disease"].str.lower() == disease_name.lower()]
    if disease_row.empty:
        return f" Disease '{disease_name}' not found in dataset."

    # Drop ID + Name, keep only features
    disease_features = disease_row.drop(["Disease_ID", "Disease"], axis=1).iloc[0].to_dict()

    # Convert input
    sample_X = pd.DataFrame([disease_features])
    sample_X = pd.get_dummies(sample_X)
    sample_X = sample_X.reindex(columns=X.columns, fill_value=0)

    # Predict
    pred = model.predict(sample_X)[0]

    # Apply smoothing
    pred = np.maximum(pred, 0.01)

    # Normalize
    pred = (pred / pred.sum()) * 100
    pred = np.round(pred, 2)

    # Create results
    predicted_drugs = pd.DataFrame({
        "Drug Name": drugs.set_index("Drug_ID").loc[y_pivot.columns, "Drug Name"].values,
        "Predicted_Effectiveness(%)": pred
    }).sort_values(by="Predicted_Effectiveness(%)", ascending=False)

    return predicted_drugs

# Build disease dictionary for easier input
unique_diseases = sorted(diseases['Disease'].unique())   # assuming column name is 'Disease'
disease_dict = {str(i+1): disease for i, disease in enumerate(unique_diseases)}