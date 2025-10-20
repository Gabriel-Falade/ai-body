# AI Fitness Coach - Bodyweight MVP

A computer vision-powered fitness application that analyzes bodyweight exercises in real-time, providing instant feedback on form and counting reps.

## 🎯 Features

- **11 Bodyweight Exercises** - Push-ups, Squats, Lunges, Planks, Burpees, and more
- **Real-time Analysis** - Instant rep counting and form feedback
- **Detailed Metrics** - ROM (Range of Motion), Velocity, Form Quality
- **Smart Feedback** - Get specific cues to improve your form
- **Multiple Input Modes**:
  - 🎥 **Live Camera** (local only) - Real-time analysis
  - 📁 **Upload Video** (cloud & local) - Analyze pre-recorded videos
  - 🎬 **Record & Analyze** (local only) - Record then process

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd weightlifting-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Select an exercise
   - Choose Live Camera or Upload Video
   - Start working out!

### Cloud Deployment (Streamlit Cloud)

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

**Quick Deploy:**
```bash
git add .
git commit -m "Deploy AI Fitness Coach"
git push
```

Then deploy via [share.streamlit.io](https://share.streamlit.io)

## 📋 Requirements

- **Python 3.11** (recommended)
- **Webcam** (for live camera mode)
- **Modern browser** (Chrome, Firefox, Safari)

### Python Dependencies
- streamlit >= 1.28.0
- opencv-python-headless == 4.8.1.78
- mediapipe == 0.10.8
- numpy >= 1.24.0, < 2.0.0
- protobuf == 3.20.3

### System Dependencies (Linux/Cloud)
- libgl1-mesa-glx
- libglib2.0-0
- libsm6
- libxrender1
- libxext6
- libgomp1
- ffmpeg

## 🏋️ Supported Exercises

1. **💪 Push-ups** - Depth and lockout validation
2. **🦵 Squats** - Depth tracking with knee valgus detection
3. **🚶 Lunges** - Balance and depth analysis
4. **🪵 Plank Hold** - Form quality monitoring
5. **🧘 Sit-ups** - Hip flexion ROM tracking
6. **🍑 Glute Bridges** - Hip extension analysis
7. **🦵 Donkey Kicks** - Single-leg glute activation
8. **🤸 Jumping Jacks** - Arm and leg position tracking
9. **⚡ Jump Squats** - Explosive power with landing form
10. **🏃 High Knees** - Cadence and knee height
11. **🔥 Burpees** - Full-body movement analysis

## 📊 Metrics Tracked

- **Rep Count** - Valid reps vs total attempts
- **ROM (Range of Motion)** - Percentage completion
- **Velocity** - Movement speed
- **Form Quality** - Specific feedback cues
- **Success Rate** - Percentage of valid reps
- **Real-time FPS** - Performance monitoring

## 💡 Usage Tips

### For Best Results:

1. **Positioning**
   - Stand 6-8 feet from camera
   - Ensure full body is visible
   - Use side or front view (depends on exercise)

2. **Environment**
   - Good lighting (avoid backlighting)
   - Clear background
   - Stable camera position

3. **Performance**
   - Lower resolution for faster processing
   - Disable skeleton overlay for speed boost
   - Use frame skip on slower computers

### Local vs Cloud:

**Run Locally When:**
- You want live camera feedback
- Testing with real-time analysis
- Maximum performance needed

**Use Cloud When:**
- Collecting user feedback
- Sharing with others
- Analyzing pre-recorded videos

## 🔧 Performance Optimization

### In Live Camera Mode:

1. **Open Performance Settings** (expandable panel)
2. **Adjust settings:**
   - Resolution: Low (320x240) = fastest
   - Frame Skip: 1-2 for 2-3x speedup
   - Skeleton Overlay: OFF for 20% boost

### Expected FPS:
- **Low-end laptop:** 10-15 FPS
- **Mid-range laptop:** 15-25 FPS
- **High-end desktop:** 25-30 FPS

## 🐛 Troubleshooting

### Camera not working?
- Check camera permissions in browser/OS
- Try different camera index (if multiple cameras)
- Restart the app

### Slow performance?
- Lower camera resolution
- Enable frame skipping
- Disable skeleton overlay
- Close other applications

### Cloud deployment errors?
- See [DEPLOYMENT.md](DEPLOYMENT.md)
- Check Streamlit Cloud logs
- Verify all files are committed

## 📁 Project Structure

```
weightlifting-ai/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── packages.txt               # System dependencies (Linux)
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── src/
│   ├── analysis/             # Pose analysis and metrics
│   │   ├── frame_metrics.py
│   │   ├── metrics.py
│   │   └── pose_helpers.py
│   └── exercises/            # Exercise-specific detectors
│       ├── base.py
│       ├── pushup.py
│       ├── squat.py
│       └── ... (11 total)
├── DEPLOYMENT.md             # Deployment guide
└── README.md                 # This file
```

## 🤝 Contributing

This is an MVP (Minimum Viable Product) for collecting user feedback. Contributions and feedback are welcome!

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- **MediaPipe** - Pose detection
- **OpenCV** - Computer vision
- **Streamlit** - Web framework

---

**Version:** 1.0.0 - MVP
**Status:** Active Development
**Python:** 3.11+
**Last Updated:** October 2025
