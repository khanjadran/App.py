import streamlit as st
import cv2
import numpy as np
import time
import logging
from datetime import datetime
import os
import uuid  # Import the uuid module
from face_detection import detect_faces
from monitoring import check_face_status
from logger import setup_logger, log_event
from utils import get_session_id, get_timestamp, format_elapsed_time  # Import utility functions

# Setup page configuration
st.set_page_config(
    page_title="Virtual Proctoring System",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'current_role' not in st.session_state:
    st.session_state.current_role = None
if 'student_name' not in st.session_state:
    st.session_state.student_name = ""
if 'exam_name' not in st.session_state:
    st.session_state.exam_name = ""
if 'session_id' not in st.session_state:
    st.session_state.session_id = get_session_id()
if 'warning_count' not in st.session_state:
    st.session_state.warning_count = 0
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'logs' not in st.session_state:
    st.session_state.logs = []

# Setup logger
logger = setup_logger()

def reset_session():
    "Reset all session variables for a new session"
    st.session_state.monitoring_active = False
    st.session_state.student_name = ""
    st.session_state.exam_name = ""
    st.session_state.session_id = get_session_id()
    st.session_state.warning_count = 0
    st.session_state.alerts = []
    st.session_state.start_time = None
    st.session_state.logs = []

def main():
    st.title("Virtual Proctoring System")

    # Role selection
    if st.session_state.current_role is None:
        st.header("Select Role")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Student", use_container_width=True):
                st.session_state.current_role = "student"
                st.rerun()

        with col2:
            if st.button("Invigilator", use_container_width=True):
                st.session_state.current_role = "invigilator"
                st.rerun()

    # Student interface
    elif st.session_state.current_role == "student":
        student_interface()

    # Invigilator interface
    elif st.session_state.current_role == "invigilator":
        invigilator_interface()

def student_interface():
    st.header("Student Exam Session")

    # Back button
    if st.button("Back to Role Selection"):
        st.session_state.current_role = None
        reset_session()
        st.rerun()

    # If monitoring is not active, show setup form
    if not st.session_state.monitoring_active:
        with st.form("student_info_form"):
            st.session_state.student_name = st.text_input("Your Name")
            st.session_state.exam_name = st.text_input("Exam Name")

            submitted = st.form_submit_button("Start Exam Session")
            if submitted:
                if st.session_state.student_name and st.session_state.exam_name:
                    st.session_state.monitoring_active = True
                    st.session_state.start_time = datetime.now()
                    log_event(logger, "INFO", f"Exam session started by {st.session_state.student_name} for {st.session_state.exam_name}")
                    st.session_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Session started")
                    st.rerun()
                else:
                    st.error("Please fill in all fields.")

    # If monitoring is active, show webcam feed and monitoring status
    else:
        st.write(f"Name: {st.session_state.student_name}")
        st.write(f"Exam: {st.session_state.exam_name}")

        # Calculate elapsed time
        elapsed_time = datetime.now() - st.session_state.start_time
        hours, remainder = divmod(elapsed_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        st.write(f"Elapsed Time: {hours:02d}:{minutes:02d}:{seconds:02d}")

        # Display webcam feed
        col1, col2 = st.columns([3, 1])

        with col1:
            # Create placeholder for webcam feed
            video_placeholder = st.empty()
            status_placeholder = st.empty()

            # Add option to end exam
            if st.button("End Exam"):
                st.session_state.monitoring_active = False
                log_event(logger, "INFO", f"Exam session ended by {st.session_state.student_name}")
                st.session_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Session ended")
                st.success("Exam session has ended.")
                st.rerun()

            # Run webcam capture loop
            cap = cv2.VideoCapture(0)

            try:
                while st.session_state.monitoring_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture video. Please check your webcam.")
                        break

                    # Flip horizontally for selfie view
                    frame = cv2.flip(frame, 1)

                    # Detect faces
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = detect_faces(frame_rgb)

                    # Draw rectangles around faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Check face status and display appropriate message
                    status, message, alert_level = check_face_status(faces)

                    if alert_level > 0:
                        # Log warning
                        st.session_state.warning_count += 1
                        log_msg = f"Warning: {message}"
                        log_event(logger, "WARNING", log_msg)
                        st.session_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_msg}")

                        # Add to alerts list for invigilator
                        alert = {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "student": st.session_state.student_name,
                            "message": message,
                            "level": alert_level
                        }
                        st.session_state.alerts.append(alert)

                    # Display status message
                    if alert_level == 0:
                        status_placeholder.success(message)
                    elif alert_level == 1:
                        status_placeholder.warning(message)
                    else:
                        status_placeholder.error(message)

                    # Display the frame with face detection
                    video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                    # Controls the update frequency
                    time.sleep(0.1)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                log_event(logger, "ERROR", f"Error in student monitoring: {str(e)}")

            finally:
                cap.release()

        with col2:
            st.subheader("Session Log")
            for log in st.session_state.logs:
                st.text(log)

def invigilator_interface():
    st.header("Invigilator Dashboard")

    # Back button
    if st.button("Back to Role Selection"):
        st.session_state.current_role = None
        st.rerun()

    # Display alerts and student information
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Active Students")
        if st.session_state.monitoring_active and st.session_state.student_name:
            # In a real application, this would show all active students
            # For demo, we'll show the single active student
            student_status = "🟢 Active" if st.session_state.monitoring_active else "🔴 Inactive"
            st.info(f"Student: {st.session_state.student_name} | Exam: {st.session_state.exam_name} | Status: {student_status}")

            # Show alert statistics
            st.metric("Total Warnings", st.session_state.warning_count)

            # Option to view student's camera feed
            if st.button("View Live Feed"):
                st.session_state.view_live = True
                st.rerun()
        else:
            st.info("No active students at the moment.")

    with col2:
        st.subheader("Recent Alerts")
        if st.session_state.alerts:
            for alert in reversed(st.session_state.alerts[-5:]):
                if alert["level"] == 1:
                    st.warning(f"[{alert['time']}] {alert['student']}: {alert['message']}")
                else:
                    st.error(f"[{alert['time']}] {alert['student']}: {alert['message']}")
        else:
            st.info("No alerts to display.")

    # View all alerts
    if st.session_state.alerts:
        with st.expander("View All Alerts"):
            for i, alert in enumerate(reversed(st.session_state.alerts)):
                if alert["level"] == 1:
                    st.warning(f"[{alert['time']}] {alert['student']}: {alert['message']}")
                else:
                    st.error(f"[{alert['time']}] {alert['student']}: {alert['message']}")

    # Live feed view (in a real app, this would show the selected student's feed)
    if st.session_state.get('view_live', False) and st.session_state.monitoring_active:
        st.subheader(f"Live Feed: {st.session_state.student_name}")

        # Button to close live feed
        if st.button("Close Live Feed"):
            st.session_state.view_live = False
            st.rerun()

        # Create placeholder for live feed
        live_feed_placeholder = st.empty()

        # Capture and display webcam feed (in a real app, this would be the student's feed)
        cap = cv2.VideoCapture(0)

        try:
            while st.session_state.get('view_live', False):
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video. Please check your webcam.")
                    break

                # Flip horizontally for selfie view
                frame = cv2.flip(frame, 1)

                # Detect faces
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detect_faces(frame_rgb)

                # Draw rectangles around faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Display the frame with face detection
                live_feed_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                # Controls the update frequency
                time.sleep(0.1)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            log_event(logger, "ERROR", f"Error in invigilator live feed: {str(e)}")

        finally:
            cap.release()

if __name__ == "__main__":
    main()

# Ensure the face_detection, monitoring, logger, and utils modules are in the same directory or properly installed.
# The following functions should ideally be in their respective modules (face_detection.py, monitoring.py, logger.py, utils.py).

# face_detection.py
def detect_faces(image):
    " Detect faces in an image using Haar cascades"
    # Get the path to the Haar cascade XML file (ensure it's in the correct location)
    cascade_path = "haarcascade_frontalface_default.xml"
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cascade_path)
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def detect_faces_dnn(image):
    " Alternative face detection using DNN (Deep Neural Network) model."
    # This would be implemented with a pre-trained model
    # For now, we'll use the Haar cascade as a fallback
    return detect_faces(image)

# monitoring.py
def check_face_status(faces):
    " Check the status of detected faces and determine if there are any alerts."
    num_faces = len(faces)
    if num_faces == 0:
        return "no_face", "No face detected. Please ensure your face is visible.", 2
    elif num_faces == 1:
        return "ok", "Face detected. You are being monitored.", 0
    elif num_faces > 1:
        return "multiple_faces", f"{num_faces} faces detected. Only the exam taker should be visible.", 2
    # Default fallback
    return "ok", "Monitoring active.", 0

def calculate_attention_score(face_history, duration=30):
    " Calculate an attention score based on face detection history."
    # This would track the presence/absence of faces over time to generate a score
    # For now, we'll return a placeholder value
    return 1.0

# logger.py
def setup_logger():
    " Set up a logger for the application."
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    # Create a logger
    logger = logging.getLogger("proctoring_system")
    logger.setLevel(logging.INFO)
    # Create a file handler
    log_file = f"logs/proctoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def log_event(logger, level, message):
    " Log an event with the specified level and message."
    if level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    else:
        logger.info(message)

# utils.py
def get_session_id():
    " Generate a unique session ID."
    return str(uuid.uuid4())

def get_timestamp():
    " Get current timestamp in a formatted string."
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def format_elapsed_time(seconds):
    " Format seconds as HH:MM:SS."
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

# .streamlit/config.toml (create this file in the same directory as your Streamlit app)
# [server]
# headless = true
# address = "0.0.0.0"
# port = 5000
#
# [theme]
# primaryColor = "#1f77b4"
# backgroundColor = "#ffffff"
# secondaryBackgroundColor = "#f0f2f6"
# textColor
