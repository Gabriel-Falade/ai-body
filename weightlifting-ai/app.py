# app.py - Simplified AI Fitness Coach MVP

import os
import sys
import time
import tempfile
from typing import List, Optional, Dict, Any
import cv2 as cv
import mediapipe as mp
import numpy as np
import streamlit as st

# --- Make 'src' importable when running from project root ---
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Project imports
from analysis.pose_helpers import mp_results_to_dict
from analysis.frame_metrics import compute_frame_metrics, FrameMetricsState

# Import all exercise detectors
from exercises.pushup import PushUpDetector
from exercises.jumping_jacks import JumpingJacksDetector
from exercises.squat import SquatDetector
from exercises.burpee import BurpeeDetector
from exercises.donkey_kicks import DonkeyKicksDetector
from exercises.glute_bridge import GluteBridgeDetector
from exercises.high_knees import HighKneesCounter
from exercises.jump_squat import JumpSquatDetector
from exercises.lunge import LungeDetector
from exercises.plank import PlankMonitor
from exercises.situp import SitupDetector

# ------------------------------
# Exercise Configuration
# ------------------------------
AVAILABLE_EXERCISES = {
    "push_up": {
        "name": "Push-ups",
        "icon": "💪",
        "detector_class": PushUpDetector,
        "description": "Tracks push-up reps with depth and lockout validation",
        "metrics": ["ROM", "Velocity", "Torso Tilt"],
        "rom_key": "rom_pushup_smooth",
        "vel_key": "vel_pushup",
        "requires_lm": False,
    },
    "squat": {
        "name": "Squats",
        "icon": "🦵",
        "detector_class": SquatDetector,
        "description": "Tracks squat depth and form with knee valgus detection",
        "metrics": ["ROM", "Velocity", "Knee Valgus"],
        "rom_key": "rom_squat_smooth",
        "vel_key": "vel_squat",
        "requires_lm": False,
    },
    "jumping_jacks": {
        "name": "Jumping Jacks",
        "icon": "🤸",
        "detector_class": JumpingJacksDetector,
        "description": "Tracks jumping jacks based on arm and leg position",
        "metrics": ["ROM", "Arms Position", "Feet Position"],
        "rom_key": "rom_pushup_smooth",
        "vel_key": None,
        "requires_lm": True,
    },
    "lunge": {
        "name": "Lunges",
        "icon": "🚶",
        "detector_class": LungeDetector,
        "description": "Tracks lunge depth with balance and form checks",
        "metrics": ["ROM", "Velocity", "Balance"],
        "rom_key": "rom_squat_smooth",
        "vel_key": "vel_squat",
        "requires_lm": False,
    },
    "burpee": {
        "name": "Burpees",
        "icon": "🔥",
        "detector_class": BurpeeDetector,
        "description": "Full-body exercise with squat, push-up, and jump detection",
        "metrics": ["Squat Depth", "Push-up Depth", "Jump Height"],
        "rom_key": "rom_squat_smooth",
        "vel_key": "vel_squat",
        "requires_lm": True,
    },
    "situp": {
        "name": "Sit-ups",
        "icon": "🧘",
        "detector_class": SitupDetector,
        "description": "Core exercise tracking hip flexion ROM",
        "metrics": ["ROM", "Velocity", "Form"],
        "rom_key": "rom_squat_smooth",
        "vel_key": "vel_squat",
        "requires_lm": False,
    },
    "plank": {
        "name": "Plank Hold",
        "icon": "🪵",
        "detector_class": PlankMonitor,
        "description": "Isometric core hold with form monitoring",
        "metrics": ["Hold Time", "Form Quality", "Torso Alignment"],
        "rom_key": None,
        "vel_key": None,
        "requires_lm": False,
    },
    "glute_bridge": {
        "name": "Glute Bridges",
        "icon": "🍑",
        "detector_class": GluteBridgeDetector,
        "description": "Hip extension exercise for glute activation",
        "metrics": ["ROM", "Hip Extension", "Form"],
        "rom_key": "rom_squat_smooth",
        "vel_key": "vel_squat",
        "requires_lm": False,
    },
    "donkey_kicks": {
        "name": "Donkey Kicks",
        "icon": "🦵",
        "detector_class": DonkeyKicksDetector,
        "description": "Single-leg glute activation exercise",
        "metrics": ["Leg Height", "ROM", "Form"],
        "rom_key": None,
        "vel_key": None,
        "requires_lm": True,
    },
    "jump_squat": {
        "name": "Jump Squats",
        "icon": "⚡",
        "detector_class": JumpSquatDetector,
        "description": "Explosive squat with jump detection",
        "metrics": ["Squat Depth", "Jump Height", "Landing Form"],
        "rom_key": "rom_squat_smooth",
        "vel_key": "vel_squat",
        "requires_lm": True,
    },
    "high_knees": {
        "name": "High Knees",
        "icon": "🏃",
        "detector_class": HighKneesCounter,
        "description": "Cardio exercise counting alternating knee raises",
        "metrics": ["Knee Height", "Cadence", "Form"],
        "rom_key": None,
        "vel_key": None,
        "requires_lm": True,
    },
}

# ------------------------------
# UI helpers
# ------------------------------
def draw_info_box(img, lines: List[str], padding=10, line_height=22):
    """Bottom-right semi-opaque box with lines of text."""
    if img is None or not lines:
        return img
    h, w = img.shape[:2]
    font = cv.FONT_HERSHEY_COMPLEX
    font_scale = 0.5
    thickness = 1

    sizes = [cv.getTextSize(s, font, font_scale, thickness)[0] for s in lines]
    text_w = max((sz[0] for sz in sizes), default=0)
    box_w = text_w + padding * 2
    box_h = line_height * len(lines) + padding * 2

    x1 = max(0, w - box_w - 10)
    y1 = max(0, h - box_h - 10)
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = img.copy()
    cv.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    img = cv.addWeighted(overlay, 0.65, img, 0.35, 0)

    y = y1 + padding + 15
    for s in lines:
        cv.putText(img, s, (x1 + padding, y), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)
        y += line_height
    return img

# ------------------------------
# Workout Processor Class
# ------------------------------
class WorkoutProcessor:
    def __init__(self, exercise_key: str):
        """Initialize processor with selected exercise."""
        self.exercise_config = AVAILABLE_EXERCISES[exercise_key]
        self.exercise_key = exercise_key

        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize the appropriate detector
        detector_class = self.exercise_config["detector_class"]
        self.detector = detector_class()

        # Frame metrics state
        self.fm_state = FrameMetricsState()

        # FPS tracking
        self.fps_ema = None
        self.prev_time = time.time()

    def process_frame(self, frame, lm_dict=None, draw_skeleton=True):
        """Process a single frame."""
        # Convert BGR to RGB
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # MediaPipe pose detection
        results = self.pose.process(image_rgb)

        # Convert back to BGR
        image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
        image.flags.writeable = True

        # Get landmarks
        lm = mp_results_to_dict(results) if lm_dict is None else lm_dict

        # Calculate timing
        now = time.time()
        dt = max(1e-3, now - self.prev_time)
        self.prev_time = now
        fps_inst = 1.0 / dt
        self.fps_ema = fps_inst if self.fps_ema is None else (0.2 * fps_inst + 0.8 * self.fps_ema)

        # Compute metrics
        fm = compute_frame_metrics(lm, dt, self.fm_state)

        # Update detector based on whether it needs landmarks
        if self.exercise_config["requires_lm"]:
            rep_event, live = self.detector.update(fm, lm, now_s=now)
        else:
            rep_event, live = self.detector.update(fm, now_s=now)

        # Draw skeleton (optional for performance)
        if draw_skeleton and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # Mirror the image for natural viewing
        image = cv.flip(image, 1)

        # Top-left: reps & stage
        cv.rectangle(image, (0, 0), (240, 74), (245, 117, 16), -1)
        cv.putText(image, 'REPS', (15, 14), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(image, str(live.get("rep_count", 0)), (10, 62), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(image, 'STAGE', (88, 14), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        # Special handling for plank (shows hold time instead of stage)
        if self.exercise_key == "plank":
            held_s = live.get("held_s", 0.0) or 0.0
            cv.putText(image, f"{held_s:.1f}s", (80, 62), cv.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2, cv.LINE_AA)
        else:
            cv.putText(image, str(live.get("stage", "") or "--"), (80, 62), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

        # Bottom-right: calculations
        rom_key = self.exercise_config["rom_key"]
        vel_key = self.exercise_config["vel_key"]

        rom = live.get("rom") or (fm.get(rom_key) if rom_key else None)
        vel = live.get("vel") or (fm.get(vel_key) if vel_key else None)
        tilt = fm.get("torso_tilt_deg")

        info_lines = []
        if self.exercise_key == "plank":
            held_s = live.get("held_s", 0.0) or 0.0
            info_lines.append(f"Hold: {held_s:5.1f}s")
        else:
            if rom is not None:
                info_lines.append(f"ROM:  {rom:5.1f} %")
            if vel is not None and vel_key:
                info_lines.append(f"Vel:  {vel:5.1f} %/s")

        if tilt is not None:
            info_lines.append(f"Tilt: {tilt:4.1f} deg")
        info_lines.append(f"FPS:  {self.fps_ema:4.1f}" if self.fps_ema is not None else "FPS:  --")
        info_lines.append(f"Mode: {self.exercise_config['name']}")

        image = draw_info_box(image, info_lines)

        return image, live, rep_event

# ------------------------------
# Video Recording Function
# ------------------------------
def record_video_from_camera(duration_seconds=30):
    """Record video from camera for analysis."""
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_file.name
    temp_file.close()

    # Try to open camera
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not open camera. Please check camera permissions and try again.")
        return None

    # Get camera properties
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default fallback

    # Setup video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Recording UI
    st.info(f"🔴 Recording for {duration_seconds} seconds... Get ready!")

    video_placeholder = st.empty()
    progress_bar = st.progress(0)
    countdown_text = st.empty()

    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_time
        if elapsed > duration_seconds:
            break

        # Write frame to file
        out.write(frame)

        # Display frame
        display_frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
        video_placeholder.image(display_frame, channels="RGB", use_column_width=True)

        # Update progress
        progress = elapsed / duration_seconds
        progress_bar.progress(min(progress, 1.0))
        countdown_text.text(f"⏱️ Time remaining: {max(0, duration_seconds - int(elapsed))} seconds")

        frame_count += 1

    # Cleanup
    cap.release()
    out.release()

    if frame_count == 0:
        st.error("❌ No frames captured. Please try again.")
        os.unlink(output_path)
        return None

    st.success(f"✅ Recording complete! Captured {frame_count} frames.")
    return output_path

# ------------------------------
# Video Processing Function
# ------------------------------
def process_video(video_path, exercise_key):
    """Process video file frame by frame."""
    video_capture = cv.VideoCapture(video_path)

    if not video_capture.isOpened():
        st.error("❌ Failed to open video file")
        return

    total_frames = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv.CAP_PROP_FPS)

    if total_frames == 0:
        st.error("❌ Video has no frames")
        return

    st.info(f"📊 Processing video: {total_frames} frames at {fps:.1f} FPS")

    processor = WorkoutProcessor(exercise_key)
    frame_count = 0
    rep_events = []

    # Create placeholders
    video_placeholder = st.empty()
    progress_bar = st.progress(0)
    stats_placeholder = st.empty()

    while video_capture.isOpened():
        ok, frame = video_capture.read()
        if not ok:
            break

        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)

        # Process frame
        processed_frame, live, rep_event = processor.process_frame(frame)

        if rep_event:
            rep_events.append(rep_event)

        # Display every 2nd frame to speed up processing
        if frame_count % 2 == 0:
            display_frame = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)
            video_placeholder.image(display_frame, channels="RGB", use_column_width=True)

            # Display stats
            rom = live.get("rom", 0.0)
            stats_placeholder.markdown(f"""
            **Current Stage**: {live.get("stage", "--")}  |
            **Total Reps**: {live.get("rep_count", 0)}  |
            **ROM**: {f"{rom:.1f}%" if rom else "--"}
            """)

    video_capture.release()

    # Show summary
    final_rep_count = live.get('rep_count', 0)
    st.success(f"✅ Video processing completed!")

    # Create results columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Valid Reps", final_rep_count)
    with col2:
        st.metric("Total Attempts", len(rep_events))
    with col3:
        success_rate = (final_rep_count / len(rep_events) * 100) if rep_events else 0
        st.metric("Success Rate", f"{success_rate:.0f}%")

    # Display rep breakdown if available
    if rep_events:
        st.subheader("📊 Rep-by-Rep Breakdown")
        for i, event in enumerate(rep_events, 1):
            counted = event.get("counted", False)
            rep_class = event.get("class", "N/A")
            cues = event.get("cues", [])

            color = "🟢" if counted else "🔴"
            cues_text = ", ".join(cues) if cues else "Good form!"

            with st.expander(f"{color} Rep {i}: {rep_class.upper()}"):
                st.write(f"**Counted:** {'✅ Yes' if counted else '❌ No'}")
                st.write(f"**Feedback:** {cues_text}")

                # Show snapshot data if available
                snapshot = event.get("snapshot", {})
                if snapshot:
                    st.json(snapshot)

# ------------------------------
# Main Streamlit App
# ------------------------------
def main():
    st.set_page_config(
        layout="wide",
        page_title="AI Fitness Coach - Bodyweight Edition",
        page_icon="🏋️"
    )

    st.title("🏋️ AI Fitness Coach")
    st.markdown("### Your Personal Bodyweight Exercise Analyzer")

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    # Exercise Selection
    st.sidebar.subheader("🎯 Select Exercise")

    exercise_options = {
        key: f"{config['icon']} {config['name']}"
        for key, config in AVAILABLE_EXERCISES.items()
    }

    selected_exercise_display = st.sidebar.selectbox(
        "Choose your workout:",
        options=list(exercise_options.values()),
        index=0
    )

    # Get the exercise key from display name
    selected_exercise = [k for k, v in exercise_options.items() if v == selected_exercise_display][0]

    # Display exercise info
    exercise_config = AVAILABLE_EXERCISES[selected_exercise]
    st.sidebar.info(f"**{exercise_config['icon']} {exercise_config['name']}**\n\n{exercise_config['description']}")

    st.sidebar.markdown("---")

    # Source Selection
    st.sidebar.subheader("📹 Video Source")

    # Detect if running on Streamlit Cloud
    is_cloud = os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("STREAMLIT_SERVER_HEADLESS")

    if is_cloud:
        st.sidebar.warning("⚠️ Running on cloud - Live camera disabled. Upload videos instead!")
        source_options = ("📁 Upload Video File",)
        source = st.sidebar.radio(
            "Choose input method:",
            source_options,
            help="Upload pre-recorded workout videos for analysis"
        )
    else:
        source = st.sidebar.radio(
            "Choose input method:",
            ("🎥 Live Camera", "📁 Upload Video File", "🎬 Record & Analyze"),
            help="Live Camera: Real-time analysis (local only)\nUpload: Analyze pre-recorded video\nRecord: Record then analyze"
        )

    # ===== LIVE CAMERA MODE =====
    if source == "🎥 Live Camera":
        st.header(f"🎥 Live {exercise_config['name']} Training")

        # Camera access warning for network devices
        import socket
        hostname = socket.gethostname()
        st.info(f"📍 **Device:** {hostname}")

        st.warning("""
        ⚠️ **Important - Camera Access Requirements:**

        Camera access **ONLY works** when:
        - ✅ Accessing via `localhost` or `127.0.0.1`
        - ✅ Using HTTPS connection

        Camera access **WILL NOT work** when:
        - ❌ Accessing from another device on network (e.g., `http://192.168.x.x`)
        - ❌ Using HTTP over network

        **Solution for other devices:** Use "Upload Video File" mode instead!
        """)

        st.markdown("""
        ### 💡 Setup Tips:
        - Position yourself 6-8 feet from camera
        - Ensure your full body is visible
        - Use good lighting
        - Allow camera permissions when prompted
        """)

        # Performance settings
        with st.expander("⚙️ Performance Settings"):
            show_skeleton = st.checkbox("Show skeleton overlay", value=True, help="Disable for better performance")
            frame_skip = st.slider("Frame skip (higher = faster, less accurate)", 0, 3, 1,
                                   help="Process every Nth frame. 0=all frames, 1=every other frame, etc.")
            camera_res = st.selectbox("Camera resolution",
                                     ["Low (320x240)", "Medium (640x480)", "High (1280x720)"],
                                     index=1,
                                     help="Lower resolution = better performance")

        # Parse resolution
        res_map = {
            "Low (320x240)": (320, 240),
            "Medium (640x480)": (640, 480),
            "High (1280x720)": (1280, 720)
        }
        cam_width, cam_height = res_map[camera_res]

        # Test camera button
        st.markdown("### 📹 Step 1: Test Camera Access")
        col_test1, col_test2 = st.columns([1, 2])

        with col_test1:
            test_camera_btn = st.button("🔍 Test Camera", use_container_width=True, type="secondary")

        with col_test2:
            if test_camera_btn:
                st.write("Scanning for available cameras...")

                # Try multiple camera indices
                available_cameras = []
                for idx in range(5):
                    test_cap = cv.VideoCapture(idx)
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        test_cap.release()
                        if ret and frame is not None:
                            available_cameras.append(idx)

                if available_cameras:
                    st.success(f"✅ Found {len(available_cameras)} camera(s): {available_cameras}")
                    st.info(f"Using camera index: {available_cameras[0]}")

                    # Store the working camera index
                    if 'camera_index' not in st.session_state:
                        st.session_state.camera_index = available_cameras[0]

                    # Show camera preview
                    test_cap = cv.VideoCapture(available_cameras[0])
                    ret, frame = test_cap.read()
                    if ret:
                        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="Camera Preview", width=300)
                    test_cap.release()
                else:
                    st.error("""
                    ❌ **No cameras detected!**

                    **Common causes:**
                    1. **Windows Camera Privacy** - Windows may be blocking camera access
                       - Go to Settings → Privacy → Camera
                       - Enable "Allow apps to access your camera"
                       - Enable for Python/Terminal

                    2. **Camera in use by another app**
                       - Close: Zoom, Teams, Skype, OBS, etc.
                       - Check Task Manager for apps using camera

                    3. **Driver issues**
                       - Update camera drivers in Device Manager
                       - Restart computer

                    4. **No camera connected**
                       - Check if laptop camera is hardware-disabled
                       - External webcam plugged in?

                    **Quick fix:** Use "📁 Upload Video File" mode instead!
                    """)

        # Initialize camera index if not set
        if 'camera_index' not in st.session_state:
            st.session_state.camera_index = 0

        st.markdown("### 🏋️ Step 2: Start Workout")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            start_button = st.button("▶️ START WORKOUT", type="primary", use_container_width=True)
        with col2:
            stop_button = st.button("⏹️ STOP", type="secondary", use_container_width=True)
        with col3:
            reset_button = st.button("🔄 RESET", use_container_width=True)

        # Initialize session state
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        if 'live_processor' not in st.session_state:
            st.session_state.live_processor = None
        if 'live_rep_events' not in st.session_state:
            st.session_state.live_rep_events = []

        # Store settings in session state
        st.session_state.show_skeleton = show_skeleton
        st.session_state.frame_skip = frame_skip

        # Handle button clicks
        if start_button:
            st.session_state.camera_active = True
            st.session_state.live_processor = WorkoutProcessor(selected_exercise)
            st.session_state.live_rep_events = []

        if stop_button:
            st.session_state.camera_active = False

        if reset_button:
            st.session_state.camera_active = False
            st.session_state.live_processor = None
            st.session_state.live_rep_events = []
            st.rerun()

        # Create layout
        video_col, stats_col = st.columns([2, 1])

        with video_col:
            video_placeholder = st.empty()

        with stats_col:
            st.subheader("📊 Live Stats")
            stats_placeholder = st.empty()
            st.markdown("---")
            st.subheader("📝 Recent Reps")
            reps_placeholder = st.empty()

        # Live camera loop
        if st.session_state.camera_active:
            camera_idx = st.session_state.get('camera_index', 0)
            cap = cv.VideoCapture(camera_idx)

            # Try DirectShow backend on Windows for better compatibility
            if not cap.isOpened():
                cap = cv.VideoCapture(camera_idx, cv.CAP_DSHOW)

            if not cap.isOpened():
                st.error(f"""
                ❌ **Cannot access camera {camera_idx}!**

                **Troubleshooting Steps:**

                1. **Click "🔍 Test Camera" above** to diagnose the issue

                2. **Check Windows Camera Privacy:**
                   - Press Win+I → Privacy & Security → Camera
                   - Turn ON "Camera access"
                   - Turn ON "Let apps access your camera"
                   - Scroll down and enable for "Desktop apps"

                3. **Close other apps using camera:**
                   - Zoom, Teams, Skype, Discord, OBS
                   - Press Ctrl+Shift+Esc to open Task Manager
                   - End any video apps

                4. **Restart your app:**
                   - Close this browser tab
                   - Stop Streamlit (Ctrl+C in terminal)
                   - Run again: `streamlit run app.py`

                5. **Try a different browser:**
                   - Chrome, Firefox, or Edge

                **Still not working?** Use "📁 Upload Video File" mode instead!
                """)
                st.session_state.camera_active = False
            else:
                # Camera settings for better performance
                cap.set(cv.CAP_PROP_FRAME_WIDTH, cam_width)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, cam_height)
                cap.set(cv.CAP_PROP_FPS, 30)
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag

                frame_count = 0
                max_frames = 300  # Process up to 300 frames before requiring rerun

                while st.session_state.camera_active and frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Failed to read from camera")
                        break

                    frame_count += 1

                    # Skip frames based on performance setting
                    skip = st.session_state.frame_skip
                    if skip > 0 and frame_count % (skip + 1) != 0:
                        continue

                    # Process frame with skeleton option
                    processed_frame, live, rep_event = st.session_state.live_processor.process_frame(
                        frame,
                        draw_skeleton=st.session_state.show_skeleton
                    )

                    # Store rep events
                    if rep_event:
                        st.session_state.live_rep_events.append(rep_event)

                    # Display processed frame
                    display_frame = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)
                    video_placeholder.image(display_frame, channels="RGB", use_column_width=True)

                    # Update stats (handle None values)
                    fps = st.session_state.live_processor.fps_ema or 0.0

                    if selected_exercise == "plank":
                        held_s = live.get('held_s') or 0.0
                        stats_placeholder.markdown(f"""
                        **Hold Time:** {held_s:.1f}s
                        **Status:** {'✅ Good Form' if live.get('hold_ok') else '⚠️ Check Form'}
                        **FPS:** {fps:.1f}
                        """)
                    else:
                        rom = live.get('rom') or 0.0
                        stats_placeholder.markdown(f"""
                        **Reps:** {live.get('rep_count', 0)}
                        **Stage:** {live.get('stage', '--')}
                        **ROM:** {rom:.1f}%
                        **FPS:** {fps:.1f}
                        """)

                    # Show recent reps (update less frequently for performance)
                    if frame_count % 5 == 0 and st.session_state.live_rep_events:
                        recent_reps = st.session_state.live_rep_events[-5:]
                        reps_text = ""
                        for event in reversed(recent_reps):
                            color = "🟢" if event.get("counted") else "🔴"
                            cues = event.get('cues', ['Good!'])
                            cue_text = cues[0] if cues else "Good!"
                            reps_text += f"{color} {event.get('class', 'N/A')}: {cue_text}\n\n"
                        reps_placeholder.markdown(reps_text)

                    # No sleep needed - let it run as fast as possible

                cap.release()

                # Auto-rerun to continue loop
                if st.session_state.camera_active:
                    time.sleep(0.1)
                    st.rerun()
        else:
            video_placeholder.info("📹 Click START to begin live analysis")

            # Show summary if we have rep events
            if st.session_state.live_rep_events:
                st.markdown("---")
                st.subheader("📊 Session Summary")

                total_reps = sum(1 for e in st.session_state.live_rep_events if e.get('counted'))
                total_attempts = len(st.session_state.live_rep_events)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Valid Reps", total_reps)
                with col2:
                    st.metric("Total Attempts", total_attempts)
                with col3:
                    success_rate = (total_reps / total_attempts * 100) if total_attempts else 0
                    st.metric("Success Rate", f"{success_rate:.0f}%")

    # ===== VIDEO UPLOAD MODE =====
    elif source == "📁 Upload Video File":
        st.header(f"📁 {exercise_config['name']} Video Analysis")

        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            help="Supported formats: MP4, MOV, AVI, MKV, WEBM"
        )

        if uploaded_file is None:
            st.info("👆 Please upload a video file to begin analysis")

            if is_cloud:
                st.markdown("""
                ### 📱 How to Record Videos for Upload:

                1. **Use your phone camera** or laptop webcam to record
                2. **Position yourself** 6-8 feet from camera
                3. **Ensure full body is visible** in the frame
                4. **Record 10-30 seconds** of your exercise
                5. **Upload the video** using the file uploader above

                **Supported formats:** MP4, MOV, AVI, MKV, WEBM
                """)

            # Show example of what metrics will be tracked
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📊 Metrics Tracked:")
                for metric in exercise_config['metrics']:
                    st.write(f"✓ {metric}")

            with col2:
                st.markdown("### 💡 Tips for Best Results:")
                st.write("✓ Full body visible in frame")
                st.write("✓ Good lighting")
                st.write("✓ Clear background")
                st.write("✓ Camera 6-8 feet away")
                st.write("✓ Side or front view")

            return

        st.success("✅ File uploaded successfully!")

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # Process button
        if st.button("▶️ Analyze Video", type="primary", use_container_width=True):
            try:
                process_video(video_path, selected_exercise)
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                import traceback
                with st.expander("Show Error Details"):
                    st.code(traceback.format_exc())
            finally:
                if os.path.exists(video_path):
                    try:
                        os.unlink(video_path)
                    except:
                        pass

    # ===== CAMERA RECORDING MODE =====
    elif source == "🎬 Record & Analyze":
        st.header(f"🎬 Record {exercise_config['name']}")

        st.markdown("""
        ### 📝 Instructions:
        1. Position yourself so your full body is visible
        2. Ensure good lighting
        3. Click 'Start Recording' below
        4. Perform your exercise
        5. Video will automatically stop after the duration
        6. Review your results!
        """)

        # Recording duration
        duration = st.slider(
            "Recording Duration (seconds)",
            min_value=10,
            max_value=60,
            value=30,
            step=5,
            help="How long to record your workout"
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔴 Start Recording", type="primary", use_container_width=True):
                video_path = record_video_from_camera(duration)

                if video_path:
                    st.markdown("---")
                    st.subheader("📊 Analyzing Your Workout...")

                    try:
                        process_video(video_path, selected_exercise)
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        import traceback
                        with st.expander("Show Error Details"):
                            st.code(traceback.format_exc())
                    finally:
                        if os.path.exists(video_path):
                            try:
                                os.unlink(video_path)
                            except:
                                pass

        with col2:
            st.info(f"**Selected Exercise:** {exercise_config['icon']} {exercise_config['name']}\n\n**Duration:** {duration}s")

    # Footer
    st.sidebar.markdown("---")

    if is_cloud:
        st.sidebar.markdown("""
        ### ℹ️ About
        This AI-powered fitness coach uses computer vision to analyze your workout videos.

        **How to use on Cloud:**
        1. Record yourself doing exercises (use phone camera)
        2. Upload the video
        3. Get instant form feedback!

        **MVP Version 1.0** - Bodyweight Exercises

        💡 **Tip:** For live camera analysis, run this app locally:
        ```bash
        streamlit run app.py
        ```
        """)
    else:
        st.sidebar.markdown("""
        ### ℹ️ About
        This AI-powered fitness coach uses computer vision to analyze your form
        and count reps in real-time. Get instant feedback on your technique!

        **MVP Version 1.0** - Bodyweight Exercises
        """)

if __name__ == "__main__":
    main()
