import os
import glob
import pandas as pd

def generate_csv(folder, label2int):
    """
    Generates a CSV file containing file paths and labels for image data.

    Args:
        folder (str): Path to the dataset folder.
        label2int (dict): Mapping of label names to integers.
    """
    folder_name = os.path.basename(folder)
    labels = list(label2int.keys())
    df = pd.DataFrame(columns=["filepath", "label"])
    
    for label in labels:
        print(f"Processing: {label}")
        for filepath in glob.glob(os.path.join(folder, label, "*")):
            df = df.append({"filepath": filepath, "label": label2int[label]}, ignore_index=True)
    
    output_file = f"{folder_name}.csv"
    print(f"Saving CSV to {output_file}")
    df.to_csv(output_file, index=False)

# Example usage
generate_csv("./data/train", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})
