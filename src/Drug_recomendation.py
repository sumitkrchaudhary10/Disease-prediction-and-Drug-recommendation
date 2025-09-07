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
diseases = pd.read_excel("/content/sample_data/Disease_Dataset_Major_project.xlsx")
drugs = pd.read_excel("/content/sample_data/Drug_Dataset_Major_project.xlsx")

# # Show unique categories in both datasets
# print("🟢 Disease Categories:")
# print(diseases["Category"].unique())

# print("\n🔵 Drug Categories:")
# print(drugs["Category"].unique())

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
    #     print(f"⚠️ No drugs found for DiseaseID {disease_id} with category {disease_cat}")
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
    raise ValueError("❌ Mapping failed: No rows generated. Check category alignment!")

# Normalize percentages per disease
mapping_df["Percentage"] = mapping_df.groupby("DiseaseID")["Percentage"].transform(
    lambda x: (x / x.sum() * 100).round(2)
)

# Save for reuse
mapping_df.to_csv("disease_drug_map.csv", index=False)
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