Gesture-Controlled Calculator 🤖

A futuristic calculator powered by Computer Vision and Hand Gesture Recognition, enabling touch-free real-time calculations using OpenCV + MediaPipe.

Perform arithmetic operations simply by waving your hand in front of the camera! 🚀

📌 Features

👆 Finger Recognition (0–5) → Enter numbers

🤏 Pinch Gesture (Thumb + Index) → Confirm / Enter number

➡️⬅️⬆️⬇️ Swipe Gestures → Operators (+, −, ×, ÷)

✊ Fist (Hold) → Erase last entry

✋✋ Both Hands Open (Hold) → Clear all

⚡ Two Hands Open (Quick Tap) → Evaluate expression

🖥️ On-screen HUD display with expression, tips, and results

🔊 Optional Voice Feedback using pyttsx3

🛠 Installation

Clone the repository and install dependencies:

git clone https://github.com/your-username/gesture-calculator.git
cd gesture-calculator

# Required
pip install opencv-python mediapipe numpy

# Optional (for voice feedback)
pip install pyttsx3

▶️ Usage

Run the calculator with:

python gesture_calculator_advanced.py --voice   # voice optional

🎮 Controls (Gestures)
Gesture	Action
✋ Show fingers (0–5)	Enter number
🤏 Pinch (Thumb + Index)	Confirm number
➡️ / ⬅️ / ⬆️ / ⬇️ Swipe	Operator (+, −, ×, ÷)
✊ Fist (Hold ~0.8s)	Erase last token
✋✋ Both hands open (Hold ~1s)	Clear all
✋✋ Both hands open (Quick tap)	Evaluate
💡 Tips for Accuracy

Ensure good lighting and face the camera directly.

Hold gestures steadily for 0.1–0.4s for recognition.

Perform swipes clearly and slightly faster for correct operator detection.

🚀 Future Enhancements

🔢 Numeric keypad overlay with gesture highlight

🎶 Sci-fi sound effects + Jarvis-style voice feedback

📋 Export results to clipboard or file

🖼️ Integration with HUD tracer for multi-system gesture control

📷 Demo

(Add a GIF or screenshot of your calculator in action here)

🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open a PR or suggest improvements.

📜 License

This project is licensed under the MIT License – feel free to use and modify with attribution.
