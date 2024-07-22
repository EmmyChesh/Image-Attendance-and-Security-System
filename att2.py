import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import winsound
import logging

# logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# new attendance file for each day
def create_attendance_file():
    today = datetime.now().strftime('%Y-%m-%d')
    filename = f'Attendance_{today}.csv'
    try:
        with open(filename, 'w') as f:
            f.write('Name,Time\n')
        logging.info(f'Attendance file {filename} created.')
    except Exception as e:
        logging.error(f'Error creating attendance file: {e}')
    return filename

# mark attendance
def mark_attendance(name, filename):
    try:
        with open(filename, 'a') as f:
            now = datetime.now().strftime('%H:%M:%S')
            f.write(f'{name},{now}\n')
        logging.info(f'Attendance marked for {name}.')
    except Exception as e:
        logging.error(f'Error marking attendance: {e}')

# Load images and class names
def load_images(path):
    images = []
    class_names = []
    try:
        for cl in os.listdir(path):
            cur_img = cv2.imread(f'{path}/{cl}')
            if cur_img is not None:
                images.append(cur_img)
                class_names.append(os.path.splitext(cl)[0])
            else:
                logging.warning(f'Could not read image {cl}.')
        logging.info(f'Loaded {len(images)} images from {path}.')
    except Exception as e:
        logging.error(f'Error loading images: {e}')
    return images, class_names

# encode images with error handling
def encode_images(images):
    encode_list = []
    for img in images:
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(img_rgb)
            if face_encodings:
                encode_list.append(face_encodings[0])
            else:
                logging.warning('No face found in an image.')
        except Exception as e:
            logging.error(f'Error encoding image: {e}')
    logging.info('Encoding complete.')
    return encode_list

# play alarm sound
def play_alarm():
    try:
        frequency = 2500  # Set Frequency To 2500 Hertz
        duration = 2000  # Duration To 2000 ms == 2 second
        winsound.Beep(frequency, duration)  # Use winsound.Beep to play sound on Windows
        logging.info('Alarm played.')
    except RuntimeError as e:
        logging.error(f'Error playing alarm: {e}')

# Main function
def main():
    path = 'ImageAttendance'
    attendance_file = create_attendance_file()
    
    images, class_names = load_images(path)
    encode_list_known = encode_images(images)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error('Error: Could not open webcam.')
        return

    marked_faces = set()
    today = datetime.now().date()

    try:
        while True:
            if datetime.now().date() != today:
                today = datetime.now().date()
                attendance_file = create_attendance_file()
                marked_faces.clear()

            success, img = cap.read()
            if not success:
                logging.warning('Failed to read from webcam.')
                break

            img_s = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

            faces_cur_frame = face_recognition.face_locations(img_s)
            encodes_cur_frame = face_recognition.face_encodings(img_s, faces_cur_frame)

            for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
                matches = face_recognition.compare_faces(encode_list_known, encode_face, tolerance=0.6)
                face_dis = face_recognition.face_distance(encode_list_known, encode_face)
                match_index = np.argmin(face_dis)

                if matches[match_index]:
                    name = class_names[match_index].upper()
                    if name not in marked_faces:
                        mark_attendance(name, attendance_file)
                        marked_faces.add(name)

                    y1, x2, y2, x1 = face_loc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                else:
                    play_alarm()

            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' key to break the loop
                break
    except Exception as e:
        logging.error(f'An error occurred: {e}')
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
