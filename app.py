from flask import Flask, request, jsonify, render_template
import face_recognition
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

DATASET_DIR = "dataset"
ATT_FILE = "attendance.csv"

os.makedirs(DATASET_DIR, exist_ok=True)

def load_known_faces():
    encodings = []
    names = []
    for file in os.listdir(DATASET_DIR):
        if file.endswith(".npy"):
            encodings.append(np.load(os.path.join(DATASET_DIR, file)))
            names.append(file.replace(".npy", ""))
    return encodings, names

KNOWN_ENCODINGS, KNOWN_NAMES = load_known_faces()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/attendance")
def attendance():
    rows = []
    if os.path.exists(ATT_FILE):
        with open(ATT_FILE, "r") as f:
            for line in f:
                rows.append(line.strip().split(","))
    return render_template("attendance.html", rows=rows)

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    file = request.files.get("image")

    if not name or not file:
        return jsonify({"status": "error", "msg": "Missing data"})

    image = face_recognition.load_image_file(file)
    encs = face_recognition.face_encodings(image)

    if not encs:
        return jsonify({"status": "no_face"})

    np.save(os.path.join(DATASET_DIR, f"{name}.npy"), encs[0])

    global KNOWN_ENCODINGS, KNOWN_NAMES
    KNOWN_ENCODINGS, KNOWN_NAMES = load_known_faces()

    return jsonify({"status": "registered", "name": name})

@app.route("/mark", methods=["POST"])
def mark():
    file = request.files.get("image")
    if not file:
        return jsonify({"status": "error"})

    image = face_recognition.load_image_file(file)
    encs = face_recognition.face_encodings(image)

    if not encs:
        return jsonify({"status": "no_face"})

    face = encs[0]
    matches = face_recognition.compare_faces(KNOWN_ENCODINGS, face)
    distances = face_recognition.face_distance(KNOWN_ENCODINGS, face)

    if len(distances) == 0:
        return jsonify({"status": "unknown"})

    best = np.argmin(distances)

    if matches[best]:
        name = KNOWN_NAMES[best]
        now = datetime.now()

        with open(ATT_FILE, "a+") as f:
            f.write(f"{name},{now.strftime('%H:%M:%S')},{now.strftime('%Y-%m-%d')}\n")

        return jsonify({"status": "marked", "name": name})

    return jsonify({"status": "unknown"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
