import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import winsound

# new attendance file for each day
def create_attendance_file():
    today = datetime.now().strftime('%Y-%m-%d')
    filename = f'Attendance_{today}.csv'
    with open(filename, 'w') as f:
        f.write('Name,Time\n')
    return filename

# mark attendance
def mark_attendance(name, filename):
    with open(filename, 'a') as f:
        now = datetime.now().strftime('%H:%M:%S')
        f.write(f'{name},{now}\n')

# Load images and class names
def load_images(path):
    images = []
    class_names = []
    for cl in os.listdir(path):
        cur_img = cv2.imread(f'{path}/{cl}')
        images.append(cur_img)
        class_names.append(os.path.splitext(cl)[0])
    return images, class_names

# encode images with error handling
def encode_images(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            encode_list.append(face_encodings[0])
        else:
            print("Warning: No face found in the image.")
    return encode_list

# play alarm sound
def play_alarm():
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 2000  # Duration To 2000 ms == 2 second
    winsound.Beep(frequency, duration) 

# Main function
def main():
    path = 'ImageAttendance'
    attendance_file = create_attendance_file()
    print("Attendance file created for today.")

    images, class_names = load_images(path)
    encode_list_known = encode_images(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    captured_faces = set()
    today = datetime.now().date()

    while True:
        if datetime.now().date() != today:
            today = datetime.now().date()
            attendance_file = create_attendance_file()
            captured_faces.clear()
            print("New day, new attendance file created.")

        success, img = cap.read()
        img_s = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

        faces_cur_frame = face_recognition.face_locations(img_s)
        encodes_cur_frame = face_recognition.face_encodings(img_s, faces_cur_frame)

        for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
            matches = face_recognition.compare_faces(encode_list_known, encode_face)
            face_dis = face_recognition.face_distance(encode_list_known, encode_face)
            match_index = np.argmin(face_dis)

            if matches[match_index]:
                name = class_names[match_index].upper()
                if name not in captured_faces:
                    y1, x2, y2, x1 = face_loc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    mark_attendance(name, attendance_file)
                    captured_faces.add(name)
                    print(f"{name} marked present.")
            else:
                play_alarm()  # Play alarm if face is not recognized

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc key to break the loop
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
