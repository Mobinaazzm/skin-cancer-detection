import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

class SkinCancerDetectionApp:
    def __init__(self):
        self.model = load_model("model.h5")  # Load the pre-trained model
        self.root = tk.Tk()  
        self.root.title("Skin Cancer Detection")  
        self.setup_ui()  # Call the UI setup method
        self.root.mainloop()  # Start the Tkinter event loop

    def setup_ui(self):
        text = (
            "This is a project for detecting skin cancer.\n"
            "Please upload a photo of the damaged area according to the instructions and see the result.\n"
            "Make sure that there's no hair, clothing, or shadows in the picture!"
        )
        self.label = tk.Label(
            self.root, text=text, font=("Arial", 14), wraplength=600, justify="left"
        )
        self.label.pack(pady=10)

        # Placeholder for displaying uploaded images
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        # Button for selecting an image
        self.button = tk.Button(self.root, text="Choose Image", command=self.upload_photo)
        self.button.pack(pady=10)

        # Text box for displaying prediction results
        self.result_text = tk.Text(self.root, height=5, width=40, font=("Arial", 12))
        self.result_text.pack(pady=10)

    def upload_photo(self):
        # Open a file dialog for selecting an image
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            try:
                # Load and preprocess the image
                image = Image.open(file_path).resize((299, 299))  # Resize to model input size
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo)
                self.image_label.image = photo

                # Convert image to a numpy array and normalize
                image_array = img_to_array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

                # Make a prediction
                prediction = self.model.predict(image_array)
                result = "Malignant" if prediction[0] < 0.5 else "Benign"

                # Display the result
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Prediction: {result}")

            except Exception as e:
                # Handle errors 
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Error: {e}")

if __name__ == "__main__":
    SkinCancerDetectionApp()

