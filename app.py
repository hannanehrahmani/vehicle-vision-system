import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Import custom modules
from src.detector import ObjectDetector, LicensePlateDetector
from src.lane_detection import LaneDetector
from src.traffic_sign import TrafficSignDetector
from src.alerts import AlertSystem
from utils.video_processor import VideoProcessor, WebcamProcessor
from utils.visualization import Visualizer, StatisticsVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="🚗 Vehicle Vision System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 0rem;
    }
    .alert-high {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0px;
    }
    .alert-medium {
        background-color: #ffaa00;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = ObjectDetector("models/best.pt") if Path("models/best.pt").exists() else None

if 'lane_detector' not in st.session_state:
    st.session_state.lane_detector = LaneDetector()

if 'traffic_detector' not in st.session_state:
    st.session_state.traffic_detector = TrafficSignDetector()

if 'alert_system' not in st.session_state:
    st.session_state.alert_system = AlertSystem()

if 'license_plate_detector' not in st.session_state:
    st.session_state.license_plate_detector = LicensePlateDetector("models/license_plate.pt") if Path("models/license_plate.pt").exists() else None

# Header
st.title("🚗 Vehicle Vision System")
st.markdown("### YOLO-based Autonomous Vehicle Detection with Streamlit")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    st.subheader("📦 Model Selection")
    model_choice = st.selectbox(
        "Select Detection Model:",
        ["best.pt", "yolov8n.pt", "yolov8s.pt", "model.pt"],
        key="model_select"
    )
    
    # Detection threshold
    st.subheader("🎯 Detection Settings")
    conf_threshold = st.slider("Confidence Threshold:", 0.0, 1.0, 0.5, 0.05)
    
    # Enable/Disable features
    st.subheader("✨ Features")
    enable_lane_detection = st.checkbox("🛣️ Lane Detection", value=True)
    enable_traffic_signs = st.checkbox("🚦 Traffic Sign Detection", value=True)
    enable_license_plate = st.checkbox("📋 License Plate Detection", value=True)
    enable_alerts = st.checkbox("🔔 Alert System", value=True)
    
    st.markdown("---")
    st.info("💡 Tip: Upload an image or video, or use your webcam for real-time detection!")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📷 Image Detection", "🎬 Video Detection", "📹 Webcam Detection", "📊 Statistics", "ℹ️ About"]
)

# TAB 1: Image Detection
with tab1:
    st.header("📷 Image Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_image is not None:
            # Read image
            image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run detection
            if st.session_state.detector:
                st.info("🔄 Processing image...")
                
                results = st.session_state.detector.detect(image)
                detections = st.session_state.detector.get_detections_info(results)
                
                output_image = st.session_state.detector.draw_boxes(image, results)
                output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                
                # Lane detection
                if enable_lane_detection:
                    lanes = st.session_state.lane_detector.detect_lanes(image)
                    output_image_rgb = cv2.cvtColor(
                        st.session_state.lane_detector.draw_detected_lanes(output_image, lanes),
                        cv2.COLOR_BGR2RGB
                    )
                
                # Traffic sign detection
                if enable_traffic_signs:
                    signs = st.session_state.traffic_detector.detect_signs(image)
                    output_image = st.session_state.traffic_detector.draw_signs(output_image, signs)
                    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                
                # License plate detection
                if enable_license_plate and st.session_state.license_plate_detector:
                    plates = st.session_state.license_plate_detector.detect(image)
                    output_image = st.session_state.license_plate_detector.draw_plates(output_image, plates)
                    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                
                st.success("✅ Detection complete!")
            else:
                st.error("❌ Model not loaded. Please check models folder.")
    
    with col2:
        if uploaded_image is not None and st.session_state.detector:
            st.subheader("Detection Results")
            st.image(output_image_rgb, use_column_width=True)
            
            # Display statistics
            st.subheader("📊 Detection Statistics")
            if detections:
                st.write(f"**Total Objects Detected:** {len(detections)}")
                
                with st.expander("📋 Detailed Detections"):
                    for i, det in enumerate(detections):
                        st.write(f"**Detection {i+1}:**")
                        st.write(f"  - Class: {det['class_id']}")
                        st.write(f"  - Confidence: {det['confidence']:.2%}")
                        st.write(f"  - Position: ({det['x1']}, {det['y1']}) to ({det['x2']}, {det['y2']})")
            else:
                st.info("No objects detected in this image.")

# TAB 2: Video Detection
with tab2:
    st.header("🎬 Video Detection")
    
    uploaded_video = st.file_uploader("Upload a video:", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video is not None:
        # Save video temporarily
        video_path = f"temp_video_{datetime.now().timestamp()}.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
        
        # Process video
        video_processor = VideoProcessor(video_path)
        video_processor.open_video(video_path)
        
        st.info(f"📹 Video Info: {video_processor.frame_count} frames @ {video_processor.fps} FPS")
        
        # Frame selection
        frame_number = st.slider("Select Frame:", 0, video_processor.frame_count - 1, 0)
        
        frame = video_processor.get_frame_at(frame_number)
        
        if frame is not None and st.session_state.detector:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Frame")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            with col2:
                st.subheader("Detection Results")
                
                # Run detection
                results = st.session_state.detector.detect(frame)
                detections = st.session_state.detector.get_detections_info(results)
                
                output_frame = st.session_state.detector.draw_boxes(frame, results)
                
                # Lane detection
                if enable_lane_detection:
                    lanes = st.session_state.lane_detector.detect_lanes(frame)
                    output_frame = st.session_state.lane_detector.draw_detected_lanes(output_frame, lanes)
                
                # Traffic signs
                if enable_traffic_signs:
                    signs = st.session_state.traffic_detector.detect_signs(frame)
                    output_frame = st.session_state.traffic_detector.draw_signs(output_frame, signs)
                
                st.image(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.write(f"**Objects Detected:** {len(detections)}")
        
        video_processor.close()
        
        # Clean up
        import os
        if os.path.exists(video_path):
            os.remove(video_path)

# TAB 3: Webcam Detection
with tab3:
    st.header("📹 Webcam Detection")
    
    webcam_enabled = st.checkbox("Enable Webcam", value=False)
    
    if webcam_enabled:
        st.warning("⚠️ Webcam feature requires local setup with OpenCV.")
        st.info("Run this app locally to use webcam functionality.")

# TAB 4: Statistics
with tab4:
    st.header("📊 Statistics & Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Alert Summary")
        alert_summary = st.session_state.alert_system.get_alert_summary()
        
        for alert_type, count in alert_summary.items():
            st.metric(alert_type.replace('_', ' '), count)
    
    with col2:
        st.subheader("Latest Alerts")
        latest_alerts = st.session_state.alert_system.get_latest_alerts(5)
        
        if latest_alerts:
            for alert in latest_alerts:
                severity_color = "🔴" if alert['severity'] == 'HIGH' else "🟠"
                st.write(f"{severity_color} **{alert['type']}**: {alert['message']}")
                st.caption(f"Time: {alert['timestamp'].strftime('%H:%M:%S')}")
        else:
            st.info("No alerts recorded yet.")

# TAB 5: About
with tab5:
    st.header("ℹ️ About Vehicle Vision System")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    The Vehicle Vision System is a comprehensive autonomous vehicle perception module that uses:
    - **YOLO v8** for object detection
    - **Lane detection** algorithms for road tracking
    - **Traffic sign recognition** for sign detection
    - **License plate detection** for vehicle identification
    - **Alert system** for safety warnings
    
    ## 🔧 Features
    
    - 🚗 **Object Detection**: Detect vehicles, pedestrians, and obstacles
    - 🛣️ **Lane Detection**: Track road lanes and detect lane departures
    - 🚦 **Traffic Signs**: Recognize traffic signs and road markers
    - 📋 **License Plates**: Detect and extract license plate information
    - 🔔 **Alert System**: Real-time collision and safety warnings
    - 📊 **Analytics**: Track and analyze detection statistics
    
    ## 📦 Models Used
    
    - **YOLOv8n.pt**: Nano model (fast, low resources)
    - **YOLOv8s.pt**: Small model (balanced)
    - **best.pt**: Custom trained model (high accuracy)
    - **model.pt**: Custom model variant
    - **license_plate.pt**: License plate detection model
    
    ## 🚀 Getting Started
    
    1. Upload an image or video
    2. Configure detection settings
    3. Enable desired features
    4. View results and statistics
    
    ## 📝 Requirements
    
    - Python 3.8+
    - OpenCV
    - PyTorch
    - YOLO v8
    - Streamlit
    
    ## 👨‍💻 Author
    
    **Hana** - Vehicle Vision System Developer
    
    ---
    
    **Made with ❤️ for Autonomous Vehicles**
    """)
    
    st.markdown("---")
    
    st.subheader("🔗 Resources")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("[📚 YOLO Documentation](https://docs.ultralytics.com/)")
    
    with col2:
        st.markdown("[🎬 OpenCV Documentation](https://opencv.org/)")
    
    with col3:
        st.markdown("[🌐 Streamlit Documentation](https://docs.streamlit.io/)")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    Vehicle Vision System v1.0 | Made with Streamlit | © 2024
</div>
""", unsafe_allow_html=True)
