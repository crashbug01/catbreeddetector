import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd
from knn import knn

def extract_average_color(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or path is incorrect: {image_path}")
    
    average_color_per_row = np.mean(image, axis=0)
    average_color = np.mean(average_color_per_row, axis=0)
    return average_color[::-1]

def load_dataset(file_path):
    df = pd.read_excel(file_path)
    X = df[['Average Red', 'Average Green', 'Average Blue']].values
    y = df['Label'].values
    return X, y

class CatBreedDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat Breed Detector")
        
        # Set window size to a larger dimension
        window_width = 800  # Width in pixels
        window_height = 680  # Height in pixels
        self.root.geometry(f"{window_width}x{window_height}")  # Set the window size
        
        self.setup_ui()
        
        self.image_path = None
        self.roi_coordinates = None
        self.X_train, self.y_train = load_dataset("Feature.xlsx")
    
    def setup_ui(self):
        # Frame for Input Image
        self.frame_input = tk.Frame(self.root, bd=2, relief=tk.SUNKEN, width=350, height=350)
        self.frame_input.grid(row=0, column=0, padx=20, pady=20, sticky='nsew')
        self.frame_input.pack_propagate(False)  # Prevent frame from resizing to fit the content
        tk.Label(self.frame_input, text="Frame Inputan").pack(pady=10)
        self.image_label = tk.Label(self.frame_input)
        self.image_label.pack(padx=10, pady=10)

        # Frame for ROI
        self.frame_roi = tk.Frame(self.root, bd=2, relief=tk.SUNKEN, width=350, height=350)
        self.frame_roi.grid(row=0, column=1, padx=20, pady=20, sticky='nsew')
        self.frame_roi.pack_propagate(False)  # Prevent frame from resizing to fit the content
        tk.Label(self.frame_roi, text="Frame Koordinat ROI").pack(pady=10)
        self.roi_label = tk.Label(self.frame_roi)
        self.roi_label.pack(padx=10, pady=10)
        
        # Buttons
        self.upload_button = tk.Button(self.root, text="Input Gambar", command=self.upload_image, bg='#00A69B', fg='white')
        self.upload_button.grid(row=1, column=0, padx=20, pady=10)
        
        self.set_roi_button = tk.Button(self.root, text="Atur ROI", command=self.set_roi, bg='#00A69B', fg='white')
        self.set_roi_button.grid(row=1, column=1, padx=20, pady=10)
        
        self.predict_button = tk.Button(self.root, text="Prediksi", command=self.predict_breed, bg='#00A69B', fg='white')
        self.predict_button.grid(row=2, column=0, columnspan=2, pady=20)
        
        # Result Section
        self.result_roi_label = tk.Label(self.root, text="Koordinat ROI: ", font=('Poppins', 10))
        self.result_roi_label.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.result_label = tk.Label(self.root, text="Hasil Prediksi:", font=("Helvetica", 16))
        self.result_label.grid(row=4, column=0, columnspan=2, pady=10)
        
        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_app, bg='#D40000', fg='white')
        self.reset_button.grid(row=10, column=0, columnspan=2, pady=20)
    
    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if self.image_path:
            self.display_image(self.image_path, self.image_label)
            messagebox.showinfo("Image Selected", f"Selected image: {self.image_path}")
    
    def display_image(self, image_path, label):
        image = Image.open(image_path)
        image = image.resize((350, 350), Image.LANCZOS)
        self.image_tk = ImageTk.PhotoImage(image)
        label.config(image=self.image_tk)
    
    def set_roi(self):
        if self.image_path:
            image = cv2.imread(self.image_path)
            self.roi_coordinates = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
            self.result_roi_label.config(text=f'Koordinat ROI: {self.roi_coordinates}')
        else:
            messagebox.showerror("Error", "Please upload an image first.")
    
    def predict_breed(self):
        if self.image_path and self.roi_coordinates:
            x, y, w, h = self.roi_coordinates
            image = cv2.imread(self.image_path)
            roi_image = image[y:y+h, x:x+w]
            roi_img = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
            roi_img = roi_img.resize((350, 350), Image.LANCZOS)
            self.roi_tk = ImageTk.PhotoImage(roi_img)
            self.roi_label.config(image=self.roi_tk)
            
            prediction = self.predict_breed_from_roi(roi_image)
            self.result_label.config(text=f'Hasil Prediksi: {prediction}')
        else:
            messagebox.showerror("Error", "Please upload an image and set ROI first.")
    
    def predict_breed_from_roi(self, roi_image):
        avg_red = np.mean(roi_image[:, :, 0])
        avg_green = np.mean(roi_image[:, :, 1])
        avg_blue = np.mean(roi_image[:, :, 2])
        
        test_features = np.array([[avg_red, avg_green, avg_blue]])
        clf = knn(k=5)
        clf.fit(self.X_train, self.y_train)
        prediction = clf.predict(test_features)
        
        return prediction[0]
    
    def reset_app(self):
        self.image_label.config(image='')
        self.roi_label.config(image='')
        self.result_label.config(text="Hasil Prediksi:")
        self.result_roi_label.config(text="Koordinat ROI:")
        self.image_path = None
        self.roi_coordinates = None

if __name__ == "__main__":
    root = tk.Tk()
    app = CatBreedDetectorApp(root)
    root.mainloop()
