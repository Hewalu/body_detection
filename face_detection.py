import cv2
import os


ENABLE_FACE_DETECTION = True
ENABLE_EYE_DETECTION = True
ENABLE_SMILE_DETECTION = True
ENABLE_PROFILE_FACE_DETECTION = False
ENABLE_FULL_BODY_DETECTION = False
ENABLE_UPPER_BODY_DETECTION = False
ENABLE_LOWER_BODY_DETECTION = False


CASCADE_DIR = 'haarcascade'
face_cascade_path = os.path.join(CASCADE_DIR, 'haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join(CASCADE_DIR, 'haarcascade_eye.xml')
smile_cascade_path = os.path.join(CASCADE_DIR, 'haarcascade_smile.xml')
profile_face_cascade_path = os.path.join(CASCADE_DIR, 'haarcascade_profileface.xml')
full_body_cascade_path = os.path.join(CASCADE_DIR, 'haarcascade_fullbody.xml')
upper_body_cascade_path = os.path.join(CASCADE_DIR, 'haarcascade_upperbody.xml')
lower_body_cascade_path = os.path.join(CASCADE_DIR, 'haarcascade_lowerbody.xml')


COLOR_FACE = (255, 0, 0)
COLOR_EYE = (0, 255, 0)
COLOR_SMILE = (255, 0, 255)
COLOR_PROFILE_FACE = (255, 255, 0)
COLOR_FULL_BODY = (0, 0, 255)
COLOR_UPPER_BODY = (0, 255, 255)
COLOR_LOWER_BODY = (255, 255, 255)


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1
TEXT_OFFSET_Y = -10


cascades = {}
cascade_paths = {
    'face': (face_cascade_path, ENABLE_FACE_DETECTION),
    'eye': (eye_cascade_path, ENABLE_EYE_DETECTION),
    'smile': (smile_cascade_path, ENABLE_SMILE_DETECTION),
    'profile_face': (profile_face_cascade_path, ENABLE_PROFILE_FACE_DETECTION),
    'full_body': (full_body_cascade_path, ENABLE_FULL_BODY_DETECTION),
    'upper_body': (upper_body_cascade_path, ENABLE_UPPER_BODY_DETECTION),
    'lower_body': (lower_body_cascade_path, ENABLE_LOWER_BODY_DETECTION),
}

for name, (path, enabled) in cascade_paths.items():
    if enabled:
        if not os.path.exists(path):
            print(f"Fehler: Die Kaskaden-Datei für '{name}' wurde nicht gefunden unter: {path}")
            print("Bitte stellen Sie sicher, dass die Datei im 'haarcascade'-Verzeichnis liegt.")

            cascade_paths[name] = (path, False)
        else:
            cascades[name] = cv2.CascadeClassifier(path)
            if cascades[name].empty():
                print(f"Fehler: Konnte die Kaskade für '{name}' nicht laden von: {path}")
                cascade_paths[name] = (path, False)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    exit()

print("Starte Erkennung... Drücke 'q' zum Beenden.")


window_name = 'Objekterkennung'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# cv2.resizeWindow(window_name, 1280, 720)

while True:

    ret, frame = cap.read()
    if not ret:
        print("Fehler: Frame konnte nicht gelesen werden.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    detected_faces = []

    if cascade_paths['face'][1] and 'face' in cascades:
        faces = cascades['face'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_FACE, 2)
            cv2.putText(frame, 'Gesicht', (x, y + TEXT_OFFSET_Y), FONT, FONT_SCALE, COLOR_FACE, FONT_THICKNESS)
            detected_faces.append((x, y, w, h))

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            if cascade_paths['eye'][1] and 'eye' in cascades:
                eyes = cascades['eye'].detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=7, minSize=(25, 25))
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), COLOR_EYE, 2)
                    cv2.putText(roi_color, 'Auge', (ex, ey + TEXT_OFFSET_Y), FONT, FONT_SCALE, COLOR_EYE, FONT_THICKNESS)

            if cascade_paths['smile'][1] and 'smile' in cascades:

                smiles = cascades['smile'].detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), COLOR_SMILE, 2)
                    cv2.putText(roi_color, 'Laecheln', (sx, sy + TEXT_OFFSET_Y), FONT, FONT_SCALE, COLOR_SMILE, FONT_THICKNESS)

    if cascade_paths['profile_face'][1] and 'profile_face' in cascades:
        profile_faces = cascades['profile_face'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        for (px, py, pw, ph) in profile_faces:

            is_overlapping = False
            for (fx, fy, fw, fh) in detected_faces:
                if px < fx + fw and px + pw > fx and py < fy + fh and py + ph > fy:
                    is_overlapping = True
                    break
            if not is_overlapping:
                cv2.rectangle(frame, (px, py), (px+pw, py+ph), COLOR_PROFILE_FACE, 2)
                cv2.putText(frame, 'Profil', (px, py + TEXT_OFFSET_Y), FONT, FONT_SCALE, COLOR_PROFILE_FACE, FONT_THICKNESS)

    if cascade_paths['full_body'][1] and 'full_body' in cascades:
        full_bodies = cascades['full_body'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 100))
        for (bx, by, bw, bh) in full_bodies:
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), COLOR_FULL_BODY, 2)
            cv2.putText(frame, 'Koerper', (bx, by + TEXT_OFFSET_Y), FONT, FONT_SCALE, COLOR_FULL_BODY, FONT_THICKNESS)

    if cascade_paths['upper_body'][1] and 'upper_body' in cascades:
        upper_bodies = cascades['upper_body'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
        for (ux, uy, uw, uh) in upper_bodies:
            cv2.rectangle(frame, (ux, uy), (ux+uw, uy+uh), COLOR_UPPER_BODY, 2)
            cv2.putText(frame, 'Oberkoerper', (ux, uy + TEXT_OFFSET_Y), FONT, FONT_SCALE, COLOR_UPPER_BODY, FONT_THICKNESS)

    if cascade_paths['lower_body'][1] and 'lower_body' in cascades:
        lower_bodies = cascades['lower_body'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
        for (lx, ly, lw, lh) in lower_bodies:
            cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), COLOR_LOWER_BODY, 2)
            cv2.putText(frame, 'Unterkoerper', (lx, ly + TEXT_OFFSET_Y), FONT, FONT_SCALE, COLOR_LOWER_BODY, FONT_THICKNESS)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

print("Programm beendet.")
