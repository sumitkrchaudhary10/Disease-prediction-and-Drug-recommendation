# main.py
from Disease_prediction import predict_disease, symptom_dict
from Drug_recomendation import predict_drugs_by_name, diseases

def main():
    while True:
        print("\n=== Healthcare Assistant ===")
        print("1. Predict Disease and Recommend Drugs")
        print("2. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            # Step 1: Symptom selection
            print("\nAvailable symptoms:")
            for key, val in symptom_dict.items():
                print(f"{key}: {val}")

            user_input = input("\nEnter symptom numbers (comma separated): ").strip()
            selected_symptoms = [
                symptom_dict[num.strip()]
                for num in user_input.split(",")
                if num.strip() in symptom_dict
            ]

            if not selected_symptoms:
                print("⚠️ Invalid input. Please try again.")
                continue

            # Step 2: Predict disease
            disease_prediction = predict_disease(selected_symptoms)
            print(f"\n🩺 Predicted Disease: {disease_prediction}")

            # Extract the final predicted disease name (a string)
            final_disease = disease_prediction['Final Prediction']

            # Step 3: Recommend drugs for the disease
            print(f"\n💊 Recommended Drugs for {final_disease}:")
            # Pass the extracted string to the function
            rf_pred = predict_drugs_by_name(final_disease, "rf")
            svm_pred = predict_drugs_by_name(final_disease, "svm")

            print("\nRandom Forest Prediction:\n", rf_pred)
            print("\nSVM Prediction:\n", svm_pred)
            
            # Slice the DataFrames to show only the top 2 recommendations
            print("\nRandom Forest Prediction:\n", rf_pred.head(2))
            print("\nSVM Prediction:\n", svm_pred.head(2))

        elif choice == "2":
            print("👋 Exiting. Stay healthy!")
            break
        else:
            print("⚠️ Invalid option. Try again.")

if __name__ == "__main__":
    main()