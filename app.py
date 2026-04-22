import os
import json
import cv2
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from utils.logger import log_result
from ultralytics import YOLO

# -------------------------------------------------
# App Setup
# -------------------------------------------------
load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"]          = os.getenv("SECRET_KEY", "dev-secret-key")
app.config["UPLOAD_FOLDER"]       = "uploads"
app.config["RESULTS_FOLDER"]      = "static/results"
app.config["MAX_CONTENT_LENGTH"]  = 200 * 1024 * 1024  # 200MB — fits most MP4s

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "jfif", "mp4", "mov", "avi"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)

# -------------------------------------------------
# Models
# -------------------------------------------------
road_model           = YOLO("models/yolo11s.pt")
damage_model         = YOLO("models/trained.pt")
road_condition_model = YOLO("models/YOLOv8_Small_RDD.pt")

print("[RoadModel] Classes:",          road_model.names)
print("[DamageModel] Classes:",        damage_model.names)
print("[RoadConditionModel] Classes:", road_condition_model.names)

ROAD_CLASSES = {
    "car", "truck", "bus",
    "person", "rider",
    "bike", "motor",
    "traffic light green",
    "traffic light red",
    "traffic light yellow",
    "traffic light",
    "traffic sign",
    "train"
}

ROAD_CONDITION_CLASSES = {
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Pothole"
}

DAMAGE_CONF_THRESHOLD         = 0.40
ROAD_CONDITION_CONF_THRESHOLD = 0.40

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    return filename.rsplit(".", 1)[1].lower() in {"mp4", "mov", "avi"}


def save_annotated_image(results, prefix, filename):
    """Save a single annotated frame for image inputs."""
    annotated       = results.plot()
    result_filename = f"{prefix}_{filename}"
    result_path     = os.path.join(app.config["RESULTS_FOLDER"], result_filename)
    cv2.imwrite(result_path, annotated)
    return result_filename


def save_annotated_video(model, image_path, prefix, filename, class_filter=None, conf_threshold=0.40):
    """
    Run frame-by-frame inference on a video, write annotated output,
    and return (aggregated_detections, result_filename).
    """
    base_name       = os.path.splitext(filename)[0]
    result_filename = f"{prefix}_{base_name}.mp4"
    result_path     = os.path.join(app.config["RESULTS_FOLDER"], result_filename)

    cap = cv2.VideoCapture(image_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"avc1") 
    out    = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

    all_labels   = {}   # label → max confidence seen across frames
    frame_count  = 0
    sample_every = 2    # run inference every N frames for speed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % sample_every == 0:
            results    = model(frame, verbose=False)[0]
            annotated  = results.plot()

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label  = model.names[cls_id]
                conf   = float(box.conf[0])

                # Apply class filter if provided (road model)
                if class_filter and label not in class_filter:
                    continue

                if conf >= conf_threshold:
                    if label not in all_labels or conf > all_labels[label]:
                        all_labels[label] = round(conf, 3)
        else:
            annotated = frame   # write original frame unmodified

        out.write(annotated)

    cap.release()
    out.release()

    # Build detection list from aggregated results
    detections = [
        {"label": label, "confidence": conf}
        for label, conf in sorted(all_labels.items(), key=lambda x: -x[1])
    ]
    return detections, result_filename


# -------------------------------------------------
# Per-model detection wrappers
# -------------------------------------------------
def detect_road_objects(image_path, filename):
    if is_video(filename):
        detections, result_filename = save_annotated_video(
            road_model, image_path, "road", filename,
            class_filter=ROAD_CLASSES, conf_threshold=0.5
        )
        return detections, result_filename

    results    = road_model(image_path)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label  = road_model.names[cls_id]
        conf   = float(box.conf[0])
        if label in ROAD_CLASSES and conf >= 0.5:
            detections.append({"label": label, "confidence": round(conf, 3)})

    result_filename = save_annotated_image(results, "road", filename)
    return detections, result_filename


def detect_vehicle_damage(image_path, filename):
    if is_video(filename):
        detections, result_filename = save_annotated_video(
            damage_model, image_path, "damage", filename,
            conf_threshold=DAMAGE_CONF_THRESHOLD
        )
        damage_status  = "Damage Detected" if detections else "No Damage Detected"
        top_confidence = detections[0]["confidence"] if detections else 0.0
        return {"damage_status": damage_status, "confidence": top_confidence, "evidence": detections}, result_filename

    results    = damage_model(image_path)[0]
    detections = []
    print("=== DAMAGE MODEL RAW DETECTIONS ===")
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label  = damage_model.names[cls_id]
        conf   = float(box.conf[0])
        print(f"[DAMAGE] {label} @ {conf:.3f}")
        if conf >= DAMAGE_CONF_THRESHOLD:
            detections.append({"label": label, "confidence": round(conf, 3)})

    if detections:
        top            = max(detections, key=lambda x: x["confidence"])
        damage_status  = "Damage Detected"
        top_confidence = top["confidence"]
    else:
        damage_status  = "No Damage Detected"
        top_confidence = 0.0

    print(f"=== DAMAGE DETECTIONS KEPT: {len(detections)} ===")
    result_filename = save_annotated_image(results, "damage", filename)
    return {"damage_status": damage_status, "confidence": top_confidence, "evidence": detections}, result_filename


def detect_road_condition(image_path, filename):
    if is_video(filename):
        detections, result_filename = save_annotated_video(
            road_condition_model, image_path, "condition", filename,
            conf_threshold=ROAD_CONDITION_CONF_THRESHOLD
        )
        status         = "Road Damage Detected" if detections else "No Road Damage Detected"
        top_confidence = detections[0]["confidence"] if detections else 0.0
        return {"condition_status": status, "confidence": top_confidence, "evidence": detections}, result_filename

    results    = road_condition_model(image_path)[0]
    detections = []
    print("=== ROAD CONDITION MODEL RAW DETECTIONS ===")
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label  = road_condition_model.names[cls_id]
        conf   = float(box.conf[0])
        print(f"[ROAD CONDITION] {label} @ {conf:.3f}")
        if conf >= ROAD_CONDITION_CONF_THRESHOLD:
            detections.append({"label": label, "confidence": round(conf, 3)})

    if detections:
        top            = max(detections, key=lambda x: x["confidence"])
        status         = "Road Damage Detected"
        top_confidence = top["confidence"]
    else:
        status         = "No Road Damage Detected"
        top_confidence = 0.0

    print(f"=== ROAD CONDITION DETECTIONS KEPT: {len(detections)} ===")
    result_filename = save_annotated_image(results, "condition", filename)
    return {"condition_status": status, "confidence": top_confidence, "evidence": detections}, result_filename


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file         = request.files["file"]
        model_choice = request.form.get("model")

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Shared flag for response building
        file_is_video = is_video(filename)

        # -----------------------------
        # ROAD OBJECT DETECTION
        # -----------------------------
        if model_choice == "road":
            detections, result_filename = detect_road_objects(filepath, filename)

            log_result(
                filename=filename,
                model_name="Road Object Detection",
                detections=[
                    {
                        "label":      d["label"],
                        "verdict":    "Detected",
                        "confidence": round(d["confidence"] * 100, 2)
                    } for d in detections
                ]
            )

            return jsonify({
                "filename":      filename,
                "model_used":    "Road Object Detection",
                "detections":    detections,
                "annotated_url": f"/static/results/{result_filename}",
                "is_video":      file_is_video
            })

        # -----------------------------
        # VEHICLE DAMAGE DETECTION
        # -----------------------------
        elif model_choice == "damage":
            result, result_filename = detect_vehicle_damage(filepath, filename)

            log_result(
                filename=filename,
                model_name="Vehicle Damage Detection (YOLO11m)",
                detections=[
                    {
                        "label":      e["label"],
                        "verdict":    result["damage_status"],
                        "confidence": round(e["confidence"] * 100, 2)
                    } for e in result["evidence"]
                ] or [{"label": "vehicle", "verdict": "No Damage Detected", "confidence": 0.0}]
            )

            return jsonify({
                "filename":      filename,
                "model_used":    "Vehicle Damage Detection",
                "damage_status": result["damage_status"],
                "confidence":    result["confidence"],
                "evidence":      result["evidence"],
                "annotated_url": f"/static/results/{result_filename}",
                "is_video":      file_is_video
            })

        # -----------------------------
        # ROAD CONDITION ANALYSIS
        # -----------------------------
        elif model_choice == "road_condition":
            result, result_filename = detect_road_condition(filepath, filename)

            log_result(
                filename=filename,
                model_name="Road Condition Analysis (YOLOv8)",
                detections=[
                    {
                        "label":      e["label"],
                        "verdict":    result["condition_status"],
                        "confidence": round(e["confidence"] * 100, 2)
                    } for e in result["evidence"]
                ] or [{"label": "road", "verdict": "No Road Damage Detected", "confidence": 0.0}]
            )

            return jsonify({
                "filename":         filename,
                "model_used":       "Road Condition Analysis",
                "condition_status": result["condition_status"],
                "confidence":       result["confidence"],
                "evidence":         result["evidence"],
                "annotated_url":    f"/static/results/{result_filename}",
                "is_video":         file_is_video
            })

        else:
            return jsonify({"error": "Invalid model choice"}), 400

    except Exception as e:
        print("Upload route error:", e)
        return jsonify({"error": "Server error", "details": str(e)}), 500


@app.route("/history", methods=["GET"])
def history():
    log_path = "logs/history.json"
    if not os.path.exists(log_path):
        history_data = []
    else:
        with open(log_path, "r", encoding="utf-8") as f:
            history_data = json.load(f)
    return render_template("history.html", history=history_data)


if __name__ == "__main__":
    app.run(debug=True)