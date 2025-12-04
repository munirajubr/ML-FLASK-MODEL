import pandas as pd
import pickle


# Load the disease dataset
data = pd.read_csv("Dataset/eggplant_diseases.csv", encoding='latin1')  # update path if needed

# Optional: check structure
print("Dataset loaded successfully!")
print(data.head())


# Create a simple function to get disease information
def get_disease_info(disease_name):
    """
    Returns all matching records for the given disease name from the dataset.
    """
    result = data[data["Disease Name"].str.lower() == disease_name.lower()]
    if result.empty:
        return f"No records found for disease: {disease_name}"
    else:
        return result.to_dict(orient="records")


# Example usage
if __name__ == "__main__":
    disease = "Fusarium Wilt"  # example disease
    info = get_disease_info(disease)
    print(info)


# Save the dataset reference (optional for deployment)
pickle.dump(data, open("disease_model.pkl", "wb"))
