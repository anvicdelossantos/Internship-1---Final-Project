# Internship-1-Final-Project
This repository is made for the completion and publishing of Internship 1 Final Project at Lamina Studios, LLC.

# AI Media Processing

A local web application built with Flask that accepts image and video input and processes it through pretrained AI/ML models to perform road object detection, vehicle damage identification, and road condition analysis.

Built as a final internship project to demonstrate real-world AI/ML integration within a clean, functional web interface.

---

## Features

| Feature | Model | Classes |
|---|---|---|
| **Road Object Detection** | YOLO11s (BDD100K) | Car, Truck, Bus, Person, Rider, Bike, Motor, Traffic Lights, Traffic Sign, Train |
| **Vehicle Damage Identification** | YOLO11m (CarDD) | Dent, Scratch, Crack, Broken Lamp, Shattered Glass, Flat Tire |
| **Road Condition Analysis** | YOLOv8s (RDD2022) | Pothole, Longitudinal Crack, Transverse Crack, Alligator Crack |

- Accepts **image** (JPG, PNG, JFIF) and **video** (MP4, MOV, AVI) input
- Displays **annotated output** with bounding boxes drawn directly on the media
- Shows **confidence scores** per detection with visual progress bars
- **Processing history** log with timestamps and per-session records
- Clean two-column interface with model selection sidebar

---

## Tech Stack

- **Backend** — Python, Flask
- **AI/ML** — Ultralytics YOLO (YOLOv8 / YOLO11), OpenCV, PyTorch
- **Frontend** — HTML, CSS, Vanilla JavaScript
- **Other** — python-dotenv, Werkzeug

---

## Pretrained Models

The following pretrained model weights are required. Download each and place them in the `models/` directory.

| File | Source | Used For |
|---|---|---|
| `yolo11s.pt` | [mamounyosef/road-object-detection](https://github.com/mamounyosef/road-object-detection) | Road Object Detection |
| `trained.pt` | [ReverendBayes/YOLO11m-Car-Damage-Detector](https://github.com/ReverendBayes/YOLO11m-Car-Damage-Detector) | Vehicle Damage Identification |
| `YOLOv8_Small_RDD.pt` | [oracl4/RoadDamageDetection](https://github.com/oracl4/RoadDamageDetection) | Road Condition Analysis |

> These models are not included in this repository. Please download them directly from the sources above.

---

## Project Structure

```
ai-media-processing/
│
├── models/                        # Pretrained YOLO weights (not tracked by git)
│   ├── yolo11s.pt
│   ├── trained.pt
│   └── YOLOv8_Small_RDD.pt
│
├── static/
│   ├── style.css                  # Main stylesheet (index)
│   ├── history_style.css          # History page stylesheet
│   └── results/                   # Annotated output images/videos (auto-created)
│
├── templates/
│   ├── index.html                 # Main upload + detection interface
│   └── history.html               # Processing history page
│
├── uploads/                       # Uploaded media files (auto-created)
│
├── logs/
│   └── history.json               # Detection history log (auto-created)
│
├── utils/
│   └── logger.py                  # Logging utility
│
├── app.py                         # Main Flask application
├── .env                           # Environment variables (not tracked by git)
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/ai-media-processing.git
cd ai-media-processing
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env` and fill in your values:
```
SECRET_KEY=your-secret-key-here
```

**4. Download pretrained model weights**

Download the three `.pt` files from the sources listed in the [Pretrained Models](#pretrained-models) section and place them in the `models/` folder:
```
models/
├── yolo11s.pt
├── trained.pt
└── YOLOv8_Small_RDD.pt
```

**5. Run the application**
```bash
python app.py
```

Open your browser and go to: [http://localhost:5000](http://localhost:5000)

---

## Usage

1. **Select a model** from the left sidebar — Road Object Detection, Vehicle Damage Identification, or Road Condition Analysis
2. **Upload an image or video** using the upload area (drag and drop supported)
3. Click **Run Detection**
4. View the **annotated output** with bounding boxes and the **detection breakdown** with confidence scores
5. Visit **Processing History** to review past detections

---

## Requirements

```
flask
python-dotenv
ultralytics
opencv-python
torch
torchvision
Pillow
werkzeug
```

Install all at once:
```bash
pip install flask python-dotenv ultralytics opencv-python torch torchvision Pillow werkzeug
```

---

## Environment Variables

Create a `.env` file in the project root based on `.env.example`:

```
SECRET_KEY=your-secret-key-here
```

| Variable | Description | Required |
|---|---|---|
| `SECRET_KEY` | Flask session secret key | Yes |

---

## Notes

- Uploaded files are saved to `uploads/` and annotated results to `static/results/`. These folders are created automatically on first run.
- Detection history is stored locally in `logs/history.json`.
- Video inference runs every 2 frames for performance. Processing time scales with video length and resolution.
- The application is intended for **local use only**. Do not expose it publicly without adding proper authentication and security hardening.
- Model weights are not included in this repository and must be downloaded separately. See [Pretrained Models](#pretrained-models).

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO framework
- [mamounyosef](https://github.com/mamounyosef/road-object-detection) — Road Object Detection model trained on BDD100K
- [ReverendBayes](https://github.com/ReverendBayes/YOLO11m-Car-Damage-Detector) — Vehicle Damage Detection model
- [oracl4](https://github.com/oracl4/RoadDamageDetection) — Road Damage Detection model trained on RDD2022
- [UC Berkeley](https://bdd100k.com/) — BDD100K driving dataset
- [Crowdsensing-based Road Damage Detection Challenge (CRDDC2022)](https://github.com/sekilab/RoadDamageDetector) — RDD2022 dataset

---

## License

This project was developed as an internship final project for educational and demonstrational purposes. Pretrained model weights are subject to their respective licenses — refer to each source repository for details.
