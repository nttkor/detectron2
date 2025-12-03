# CCTV ID Recognition & Guidance System (MVP)

## 📌 Project Overview
This project is an **AI-based CCTV guidance system for the visually impaired**.
Unlike traditional smartphone-only solutions, this system processes **CCTV footage on a central PC server** to accurately identify and track users, sending real-time voice guidance to their smartphones.

### Key Features (MVP)
- **Pose-based ID Tracking**: Uses YOLOv11-Pose to assign and maintain unique IDs for users even after occlusion.
- **Robust Re-Identification**: Combined logic of Pose Similarity + Scale Smoothing + Temporal Consistency.
- **Real-time Visualization**: Displays status (NEW/CONFIRMED), ID, and Skeleton on the video feed.

---

## 📂 Repository Structure
- `samu/tracker.py`: Main execution file (PoseTracker Engine).
- `samu/tracker_design_and_plan.md`: **Project Plan & Architecture Design**.
- `samu/api_spec.yaml`: Server API Specification.
- `samu/output/`: Directory for tracking logs.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Tracker
```bash
python samu/tracker.py
```
*Note: Ensure the video path in `samu/tracker.py` matches your local environment or use a webcam (0).*

---

## 📄 Documentation
- [Project Design & Plan (PDF Source)](samu/tracker_design_and_plan.md)
- [API Specification](samu/api_spec.yaml)







