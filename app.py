import os
import cv2
import face_recognition as fr
from flask import Flask, render_template, Response, request
import recognition as rgn

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_name = None
cap = None
flag = False

img_folder = "faces/"
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

def add_face():
    global face_name, flag
    face = None
    if not cap.isOpened():
        print("cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        face = frame
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(RGB_frame, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
    try:
        path = img_folder+face_name+".jpg"
        cv2.imwrite(path, face)
        flag = True
        cap.release()
        cv2.destroyAllWindows()
    except:
        flag = False    

def mark_attendance():
    global flag
    flag = False
    face_encoding = []
    names, encodings = rgn.name_encodings()
    if not cap.isOpened():
        print("cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_loc = fr.face_locations(RGB_frame)
        face_encoding = fr.face_encodings(RGB_frame, face_loc)[0]
        matches = fr.compare_faces(encodings, face_encoding)
        if True in matches:
            flag = True
            match_index = matches.index(True)
            name = names[match_index]
        faces = face_cascade.detectMultiScale(RGB_frame, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, name, (h + 6, w - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=1)
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
    
    rgn.mark_attendense(name=name)
    cap.release()
    cv2.destroyAllWindows()
    

@app.before_request
def before_request():
    global cap 
    if request.endpoint != 'index':
        if cap is not None:
            cap.release()
            cap = None

@app.route("/capture")
def add_face_videoFeed():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
    return Response(add_face(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/capture")
def mark_attendance_videoFeed():
    global cap 
    if cap is None:
        cap = cv2.VideoCapture(0)
    return Response(mark_attendance(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def welcome():
    return render_template("index.html")

@app.route("/attendance_portal")
def choose_opr():
    return render_template("operations.html")

@app.route("/attendance_portal/add_new_face", methods=['GET', 'POST'])
def add_new_face():
    global face_name
    if request.method=='GET':
        return render_template("name.html")
    else:
        face_name = request.form['name']
        return render_template("add_face.html")

@app.route("/attendance_portal/mark_attendance")
def mark_attendance():
    return render_template("mark_attendance.html")

@app.route("/attendance_portal/add_new_face/result")
def result():
    global flag
    return render_template("success.html", flag=flag)

if __name__=="__main__":
    app.run(debug=True)