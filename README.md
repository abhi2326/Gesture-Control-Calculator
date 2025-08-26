# ✋ Gesture-Controlled Calculator 🤖  

A futuristic calculator powered by **Computer Vision** and **Hand Gesture Recognition**, enabling **touch-free real-time calculations** using **OpenCV + MediaPipe**.  

Perform arithmetic operations simply by waving your hand in front of the camera! 🚀  

---

## 📌 Features  
- 👆 **Finger Recognition (0–5)** → Enter numbers  
- 🤏 **Pinch Gesture (Thumb + Index)** → Confirm / Enter number  
- ➡️⬅️⬆️⬇️ **Swipe Gestures** → Operators (+, −, ×, ÷)  
- ✊ **Fist (Hold)** → Erase last entry  
- ✋✋ **Both Hands Open (Hold)** → Clear all  
- ⚡ **Two Hands Open (Quick Tap)** → Evaluate expression  
- 🖥️ On-screen **HUD display** with expression, tips, and results  
- 🔊 **Optional Voice Feedback** using `pyttsx3`  

---

## 🛠 Installation  

Clone the repository and install dependencies:  

```bash
git clone https://github.com/your-username/gesture-calculator.git
cd gesture-calculator

# Required
pip install opencv-python mediapipe numpy

# Optional (for voice feedback)
pip install pyttsx3
