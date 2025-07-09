# gui/app.py

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from utils.detect_nationality import load_nationality_model, predict_nationality
from utils.detect_emotion import load_emotion_model, predict_emotion
from utils.detect_age import load_age_model, predict_age
from utils.detect_dress_color import predict_dress_color

# Load all models
nationality_model = load_nationality_model("models/nationality_7class_model.pth")
emotion_model = load_emotion_model("models/emotion_detection_model.keras")
age_model = load_age_model("models/weights.28-3.73.hdf5")  # or .hdf5

# Nationality groups
INDIAN = "Indian"
AMERICAN = "White"  # adjust if your US label differs
AFRICAN = "Black"   # adjust if your African label differs

# GUI
class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Nationality Detection System")
        self.master.geometry("600x700")

        self.label = tk.Label(master, text="Upload an Image", font=("Arial", 16))
        self.label.pack(pady=10)

        self.img_label = tk.Label(master)
        self.img_label.pack(pady=10)

        self.button = tk.Button(master, text="Choose Image", command=self.upload_image)
        self.button.pack(pady=10)

        self.output = tk.Text(master, height=15, width=70)
        self.output.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        # Preview image
        img = Image.open(file_path).resize((200, 200))
        self.tk_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self.tk_img)

        # Clear previous output
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"📷 Image: {file_path}\n")

        # Run all predictions
        nationality = predict_nationality(file_path, nationality_model)
        emotion = predict_emotion(file_path, emotion_model)

        self.output.insert(tk.END, f"🌍 Nationality: {nationality}\n")
        self.output.insert(tk.END, f"😊 Emotion: {emotion}\n")

        # Apply conditional logic
        if nationality == INDIAN:
            age = predict_age(file_path, age_model)
            dress = predict_dress_color(file_path)
            self.output.insert(tk.END, f"🎂 Age: {age}\n")
            self.output.insert(tk.END, f"👕 Dress Color: {dress}\n")

        elif nationality == AMERICAN:
            age = predict_age(file_path, age_model)
            self.output.insert(tk.END, f"🎂 Age: {age}\n")

        elif nationality == AFRICAN:
            dress = predict_dress_color(file_path)
            self.output.insert(tk.END, f"👕 Dress Color: {dress}\n")

        else:
            self.output.insert(tk.END, "ℹ️ (Other nationality: No age or dress color prediction)\n")

# Run app
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
