import tkinter as tk
from tkinter import messagebox
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_IMAGE_DIR = os.path.join(BASE_DIR, "TrainingImage")
STUDENT_DETAILS_PATH = os.path.join(BASE_DIR, "StudentDetails", "StudentDetails.csv")
TRAINNER_PATH = os.path.join(TRAINING_IMAGE_DIR, "Trainner.yml")
HAARCASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
UNKNOWN_IMAGES_DIR = os.path.join(BASE_DIR, "ImagesUnknown")

# Create directories if they don't exist
os.makedirs(TRAINING_IMAGE_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(UNKNOWN_IMAGES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STUDENT_DETAILS_PATH), exist_ok=True)

class AttendanceSystem:
    def __init__(self, window):
        self.window = window
        self.window.title("Attendance System")
        self.window.configure(background='pink')
        self.window.geometry("1600x800")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create and place all UI elements
        self.create_labels()
        self.create_entries()
        self.create_buttons()
        
    def create_labels(self):
        # Main title
        tk.Label(self.window, text="ATTENDANCE MANAGEMENT PORTAL", 
                bg="black", fg="red", width=40, height=1,
                font=('Times New Roman', 35, 'bold underline')).place(x=200, y=20)
        
        # College information
        tk.Label(self.window, text="GEC College", bg="black", fg="red",
                width=20, height=2, font=('Times New Roman', 25, 'bold')).place(x=1150, y=760)
        
        # Input fields labels
        tk.Label(self.window, text="Enter Your College ID", width=20, height=2,
                fg="black", bg="red", font=('Times New Roman', 25, 'bold')).place(x=125, y=180)
        
        tk.Label(self.window, text="Enter Your Name", width=20, fg="black", 
                bg="red", height=2, font=('Times New Roman', 25, 'bold')).place(x=525, y=180)
                
        # Notification label
        tk.Label(self.window, text="NOTIFICATION", width=20, fg="black", 
                bg="red", height=2, font=('Times New Roman', 25, 'bold')).place(x=985, y=180)
        
        # Step labels
        tk.Label(self.window, text="STEP 1", width=20, fg="green", 
                bg="pink", height=2, font=('Times New Roman', 20, 'bold')).place(x=165, y=355)
        
        tk.Label(self.window, text="STEP 2", width=20, fg="green", 
                bg="pink", height=2, font=('Times New Roman', 20, 'bold')).place(x=570, y=355)
                
        tk.Label(self.window, text="STEP 3", width=20, fg="green", 
                bg="pink", height=2, font=('Times New Roman', 20, 'bold')).place(x=1025, y=342)
        
        # Attendance display label
        tk.Label(self.window, text="ATTENDANCE", width=20, fg="black", 
                bg="red", height=2, font=('Times New Roman', 30, 'bold')).place(x=120, y=470)
    
    def create_entries(self):
        # College ID entry
        self.id_entry = tk.Entry(self.window, width=30, bg="white", fg="red",
                                font=('Times New Roman', 15, 'bold'))
        self.id_entry.place(x=175, y=260)
        
        # Name entry
        self.name_entry = tk.Entry(self.window, width=30, bg="white", fg="red",
                                 font=('Times New Roman', 15, 'bold'))
        self.name_entry.place(x=575, y=260)
        
        # Notification message
        self.message = tk.Label(self.window, text="", bg="white", fg="red", 
                              width=30, height=1, font=('Times New Roman', 15, 'bold'))
        self.message.place(x=1000, y=260)
        
        # Attendance display
        self.attendance_message = tk.Label(self.window, text="", fg="red", bg="yellow",
                                          width=60, height=4, font=('times', 15, 'bold'))
        self.attendance_message.place(x=700, y=470)
    
    def create_buttons(self):
        # Image capture button
        tk.Button(self.window, text="IMAGE CAPTURE BUTTON", command=self.take_images,
                fg="white", bg="blue", width=25, height=2, 
                font=('Times New Roman', 15, 'bold')).place(x=170, y=405)
        
        # Model training button
        tk.Button(self.window, text="MODEL TRAINING BUTTON", command=self.train_images,
                fg="white", bg="blue", width=25, height=2,
                font=('Times New Roman', 15, 'bold')).place(x=570, y=405)
        
        # Attendance marking button
        tk.Button(self.window, text="ATTENDANCE MARKING BUTTON", command=self.track_images,
                fg="white", bg="red", width=30, height=3,
                font=('Times New Roman', 15, 'bold')).place(x=1000, y=392)
        
        # Quit button
        tk.Button(self.window, text="QUIT", command=self.quit_window,
                fg="white", bg="red", width=10, height=2,
                font=('Times New Roman', 15, 'bold')).place(x=700, y=625)
    
    def clear_entries(self):
        self.id_entry.delete(0, 'end')
        self.name_entry.delete(0, 'end')
        self.message.configure(text="")
    
    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False
    
    def take_images(self):
        Id = self.id_entry.get().strip()
        name = self.name_entry.get().strip()
        
        if not Id:
            self.message.configure(text="Please enter ID")
            messagebox.showwarning("Warning", "Please enter roll number properly")
            return
            
        if not name:
            self.message.configure(text="Please enter Name")
            messagebox.showwarning("Warning", "Please enter your name properly")
            return
            
        if not (self.is_number(Id) and name.isalpha()):
            if self.is_number(Id):
                self.message.configure(text="Enter Alphabetical Name")
            else:
                self.message.configure(text="Enter Numeric Id")
            return
            
        try:
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                raise RuntimeError("Could not open camera")
                
            detector = cv2.CascadeClassifier(HAARCASCADE_PATH)
            if detector.empty():
                raise RuntimeError("Could not load face detection model")
                
            sampleNum = 0
            
            while True:
                ret, img = cam.read()
                if not ret:
                    raise RuntimeError("Failed to capture image")
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                
                for (x,y,w,h) in faces:
                    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                    sampleNum += 1
                    img_path = os.path.join(TRAINING_IMAGE_DIR, f"{name}.{Id}.{sampleNum}.jpg")
                    cv2.imwrite(img_path, gray[y:y+h, x:x+w])
                    cv2.imshow('Register Face', img)
                
                if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 60:
                    break
                    
            cam.release()
            cv2.destroyAllWindows()
            
            # Save student details
            with open(STUDENT_DETAILS_PATH, 'a+', newline='') as csvFile:
                writer = csv.writer(csvFile)
                # Write header if file is empty
                if os.stat(STUDENT_DETAILS_PATH).st_size == 0:
                    writer.writerow(["Id", "Name"])
                writer.writerow([Id, name])
            
            self.message.configure(text=f"Images Saved for ID: {Id} Name: {name}")
            
        except Exception as e:
            self.message.configure(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            if 'cam' in locals() and cam.isOpened():
                cam.release()
            cv2.destroyAllWindows()
    
    def train_images(self):
        try:
            # Check if there are images to train
            if not os.listdir(TRAINING_IMAGE_DIR):
                raise ValueError("No training images found. Please capture images first.")
                
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            faces, Id = self.get_images_and_labels(TRAINING_IMAGE_DIR)
            
            if not faces:
                raise ValueError("No faces found in training images.")
                
            recognizer.train(faces, np.array(Id))
            recognizer.save(TRAINNER_PATH)
            
            self.clear_entries()
            self.message.configure(text="Image Trained")
            messagebox.showinfo('Completed', 'Model trained successfully!')
            
        except Exception as e:
            self.message.configure(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def get_images_and_labels(self, path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
        faces = []
        Ids = []
        
        for imagePath in imagePaths:
            try:
                pilImage = Image.open(imagePath).convert('L')
                imageNp = np.array(pilImage, 'uint8')
                Id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces.append(imageNp)
                Ids.append(Id)
            except Exception as e:
                print(f"Skipping invalid image {imagePath}: {str(e)}")
                continue
                
        return faces, Ids
    
    def track_images(self):
        try:
            # Check if trained model exists
            if not os.path.exists(TRAINNER_PATH):
                raise FileNotFoundError("Trainner.yml not found. Please train the model first.")
                
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(TRAINNER_PATH)
            
            # Check if face detection model exists
            if not os.path.exists(HAARCASCADE_PATH):
                raise FileNotFoundError("Haarcascade file not found.")
                
            faceCascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
            if faceCascade.empty():
                raise RuntimeError("Could not load face detection model")
            
            # Check if student details exist
            if not os.path.exists(STUDENT_DETAILS_PATH):
                raise FileNotFoundError("StudentDetails.csv not found.")
                
            df = pd.read_csv(STUDENT_DETAILS_PATH)
            if df.empty:
                raise ValueError("No student data found in StudentDetails.csv")
            
            # Initialize camera
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                raise RuntimeError("Could not open camera.")
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            col_names = ['Id', 'Name', 'Date', 'Time']
            attendance = pd.DataFrame(columns=col_names)
            
            while True:
                ret, im = cam.read()
                if not ret:
                    raise RuntimeError("Failed to capture image from camera.")
                    
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, 1.2, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
                    Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                    
                    # Check confidence level (lower is more confident)
                    if conf < 50:
                        try:
                            ts = time.time()
                            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            
                            # Find student name
                            student_info = df[df['Id'] == Id]
                            if not student_info.empty:
                                aa = student_info['Name'].values[0]
                                tt = f"{Id}-{aa}"
                                # Check if already marked today
                                if not ((attendance['Id'] == Id) & 
                                       (attendance['Date'] == date)).any():
                                    attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
                            else:
                                tt = f"Unknown ID: {Id}"
                        except Exception as e:
                            print(f"Error processing recognized face: {e}")
                            tt = "Error"
                    else:
                        Id = 'Unknown'
                        tt = str(Id)
                        if conf > 75:  # Very uncertain
                            noOfFile = len(os.listdir(UNKNOWN_IMAGES_DIR)) + 1
                            unknown_img_path = os.path.join(
                                UNKNOWN_IMAGES_DIR, f"Image{noOfFile}.jpg")
                            cv2.imwrite(unknown_img_path, im[y:y+h, x:x+w])
                    
                    cv2.putText(im, str(tt), (x, y+h), font, 1, (255, 255, 255), 2)
                
                cv2.imshow('Marking Attendance', im)
                if cv2.waitKey(1) == ord('q'):
                    break
            
            # Save attendance if any records exist
            if not attendance.empty:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H-%M-%S')
                fileName = os.path.join(
                    ATTENDANCE_DIR, f"Attendance_{date}_{timeStamp}.csv")
                attendance.to_csv(fileName, index=False)
                
                self.attendance_message.configure(text=attendance.to_string())
                self.message.configure(text="Attendance Taken")
                messagebox.showinfo('Success', 'Attendance marked successfully!')
            else:
                self.message.configure(text="No attendance marked")
                messagebox.showwarning('Warning', 'No faces recognized for attendance!')
                
        except Exception as e:
            error_msg = f"Attendance marking failed: {str(e)}"
            self.message.configure(text=error_msg)
            messagebox.showerror("Error", error_msg)
            print(error_msg)  # For debugging
        finally:
            if 'cam' in locals() and cam.isOpened():
                cam.release()
            cv2.destroyAllWindows()
    
    def quit_window(self):
        if messagebox.askyesno('Exit', 'Are you sure you want to exit?'):
            messagebox.showinfo("Thank You", "Thank you for using our software!")
            self.window.destroy()

if __name__ == "__main__":
    window = tk.Tk()
    app = AttendanceSystem(window)
    window.mainloop()