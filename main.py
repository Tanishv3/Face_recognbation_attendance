import os
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Load images and encodings
known_faces = {
   # "Tanish": face_recognition.load_image_file("photos/tanish.jpg"),
    "Lionel Messi": face_recognition.load_image_file("photos/messi.jpg"),
    "ronaldo": face_recognition.load_image_file("photos/ronaldo.jpg"),
    "mansi": face_recognition.load_image_file("photos/mansi.jpg"),  
    "Vaishnavi Pawar": face_recognition.load_image_file("photos/vaish.jpg"),
}

known_encodings = {}
for name, image in known_faces.items():
    encoding = face_recognition.face_encodings(image)[0]
    known_encodings[name] = encoding

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Initialize CSV writer
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_filename = current_date + '_attendance.csv'

# Check if file exists, if not, create a new one
file_exists = os.path.exists(csv_filename)
with open(csv_filename, 'a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    if not file_exists:
        csv_writer.writerow(['Name', 'Time'])

# Initialize attendance tracking
present_students = set()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Initialize array to store names of detected faces
    face_names = []

    # Check for matches with known faces
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(list(known_encodings.values()), face_encoding)
        name = "Unknown"

        # Check if there's a match
        if True in matches:
            matched_index = matches.index(True)
            name = list(known_encodings.keys())[matched_index]
            if name not in present_students:
                present_students.add(name)
                current_time = now.strftime("%I:%M:%p")
                with open(csv_filename, 'a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([name, current_time])

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Attendance System', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()

cv2.destroyAllWindows()