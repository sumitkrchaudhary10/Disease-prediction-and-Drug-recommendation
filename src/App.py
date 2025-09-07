# import streamlit as st
# from Disease_prediction import predict_disease, symptom_dict
# from Drug_recomendation import predict_drugs_by_name

# st.set_page_config(page_title="Disease & Drug Recommender", page_icon="💊", layout="wide")

# st.title("🩺 Disease Prediction & 💊 Drug Recommendation System")

# st.write("Select your symptoms, predict possible diseases, and get recommended drugs.")

# # ---------------------------
# # Disease Prediction Section
# # ---------------------------
# st.header("Step 1: Select Your Symptoms")

# selected_numbers = st.multiselect(
#     "Choose symptoms by number:",
#     options=list(symptom_dict.keys()),
#     format_func=lambda x: f"{x}: {symptom_dict[x]}"
# )

# if selected_numbers:
#     selected_symptoms = [symptom_dict[num] for num in selected_numbers]
#     symptom_string = ",".join(selected_symptoms)

#     predictions = predict_disease(symptom_string)

#     st.subheader("Predicted Diseases")
#     for model, pred in predictions.items():
#         st.write(f"**{model}**: {pred}")

#     final_disease = predictions["Final Prediction"]

#     # ---------------------------
#     # Drug Recommendation Section
#     # ---------------------------
#     st.header("Step 2: Recommended Drugs")

#     model_choice = st.radio("Choose Model for Drug Prediction:", ["Random Forest", "SVM"])
#     model_key = "rf" if model_choice == "Random Forest" else "svm"

#     predicted_drugs = predict_drugs_by_name(final_disease, model_choice=model_key)

#     if isinstance(predicted_drugs, str):
#         st.error(predicted_drugs)
#     else:
#         st.success(f"Recommended Drugs for **{final_disease}**")
#         st.dataframe(predicted_drugs)

# else:
#     st.info("👆 Select symptoms from the dropdown above to continue.")