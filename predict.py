#Importing necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Loading trained model
model = load_model("deepMODEmain.h5")

def predict_and_show(img_path):
    # Load and preprocess
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    label = f"🟢 Real ({confidence*100:.2f}%)" if prediction > 0.5 else f"🔴 Fake ({confidence*100:.2f}%)"

    # Show image with results
    plt.imshow(image.load_img(img_path))
    plt.axis("off")
    plt.title(label, fontsize=14, color="green" if prediction > 0.5 else "red")
    plt.show()

# Folder paths
real_folder = "dataset/real"
fake_folder = "dataset/fake"

# Loop through images in both folders
for folder, label in [(real_folder, "Real"), (fake_folder, "Fake")]:
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        if img_path.lower().endswith((".png", ".jpg", ".jpeg")):  # Taking only valid images
            print(f"Predicting: {img_path}")
            predict_and_show(img_path)
