import json
import os

KB_PATH = "data/plantvillage_kb.json"

def seed_olive():
    if not os.path.exists(KB_PATH):
        print(f"Error: {KB_PATH} not found.")
        return

    with open(KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Always update Olive data
    data["Olive"] = {
        "Peacock spot": {
            "description": "Leaves infected with Spilocaea oleagina, a fungal disease that causes black spots.",
            "source": "https://www.kaggle.com/datasets/habibulbasher01644/olive-leaf-image-dataset",
            "symptoms": "Black spots on leaves.",
            "cause": "Spilocaea oleagina (fungus)",
            "management": ""
        },
        "Aculus olearius": {
            "description": "Leaves affected by olive gall mite, which leads to deformation and discoloration.",
            "source": "https://www.kaggle.com/datasets/habibulbasher01644/olive-leaf-image-dataset",
            "symptoms": "Deformation and discoloration of leaves.",
            "cause": "Aculus olearius (mite)",
            "management": ""
        },
        "healthy": {
            "description": "Leaves without disease symptoms.",
            "source": "https://www.kaggle.com/datasets/habibulbasher01644/olive-leaf-image-dataset",
            "symptoms": "",
            "cause": "",
            "management": ""
        }
    }
    print("Updated Olive in KB with Kaggle data.")

    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    seed_olive()
