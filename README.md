# 📏 PixelScale – Real-World Object Size Measurement using Camera (CV-Based)

PixelScale is a computer vision project that allows you to **measure real-world objects using your camera**. It supports **three intelligent modes**:

1. 📐 **Manual Reference Mode** – Use a known-width object to set a reference and measure others.
2. 🌊 **Depth Estimation Mode** – Estimate dimensions using camera depth assumptions (focal length & distance).
3. 👤 **Human Face-Based Mode** – Uses average human facial proportions as scale reference (no extra tools needed!).

Perfect for engineers, DIYers, makers, or curious minds who want to **analyze object sizes in real time** through OpenCV.

---

## 🚀 Features

- ✅ Real-time object detection and measurement via webcam
- 📏 Manual reference calibration using known-size objects
- 🌐 Depth-based estimation using focal length and distance
- 👤 Proportion-based estimation using human facial features
- 📸 Save measurement frames with dimension overlays
- 🔧 Adjustable distance control for depth-based mode
- 🖥️ Clean and interactive OpenCV UI
- 🧠 Modular code structure for easy extensions

---


## 🧰 Tech Stack

- Python 3.x
- OpenCV (cv2)
- NumPy
- Haarcascade face detection

---

## 🛠️ Setup Instructions

### ✅ Prerequisites
Make sure you have Python and pip installed.

```bash
pip install opencv-python numpy
