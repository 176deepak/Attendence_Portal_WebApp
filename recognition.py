import os
import csv
from datetime import datetime
import face_recognition as fr

data_folder_path = "data/"
image_folder_path = "faces/"
filename = "face_encodings.csv"

if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)
if not os.path.exists(image_folder_path):
    os.makedirs(image_folder_path)
file_path = os.path.join(data_folder_path, filename)

def name_encodings():
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir('faces'):
        if filename.endswith('.jpg'):
            # Load the image file and convert it to RGB format
            image = fr.load_image_file('faces/' + filename)
            face_encoding = fr.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
        
            # Extract the person's name from the filename and add it to the list of known face names
            name = filename[:-5]
            known_face_names.append(name)
    return (known_face_names, known_face_encodings)  

def face_encoding(frame):
    face_loc = fr.face_locations(frame)
    encoding = fr.face_encodings(frame, face_loc)[0]
    return encoding    

def mark_attendense(name):
    curr_date_time = datetime.now()
    data = [name, curr_date_time]
    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)
