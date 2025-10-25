# 👁️ Vision Aid – Voice Controlled Object Detection & Description

A Python-based assistive vision system using **YOLOv8**, **VOSK**, and **BLIP** that:
- Detects nearby obstacles and warns about collisions.
- Responds to voice commands:
  - **“start”** → resumes obstacle detection.
  - **“stop”** → pauses detection.
  - **“describe”** → takes a snapshot and verbally describes surroundings.
- Keeps the camera window open at all times with live overlay status.

##  Setup Instructions

### 1️Clone the repository
```bash
git clone https://github.com/reinnxz/visionaid
cd visionaid
```

### 2️ Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
```

### 3️ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️ Download VOSK model
Download from:
https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip

Extract to:
~/models/vosk/vosk-model-small-en-us-0.15/
```

### 5️⃣ Run the app
```bash
python main.py
```

### Commands
- “**start**” → Begin detection  
- “**stop**” → Pause detection  
- “**describe**” → Describe surroundings  
- Press **Q** → Quit

---

Built with ❤️ using Ultralytics YOLO, Salesforce BLIP, and VOSK ASR.
