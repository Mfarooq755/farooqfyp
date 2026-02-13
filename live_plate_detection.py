import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import sqlite3
import streamlit as st

# Load YOLOv8 model
model = YOLO("best.pt")  # Replace with your YOLOv8 .pt file

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize SQLite database
conn = sqlite3.connect('car_parking.db', check_same_thread=False)
cursor = conn.cursor()

# Function to check if a plate number exists in the database
def is_car_registered(car_number):
    cursor.execute("SELECT * FROM registered_cars WHERE car_number = ?", (car_number,))
    return cursor.fetchone()

# Function to add a new plate number to the database
def register_car(car_number, owner_name):
    cursor.execute("INSERT INTO registered_cars (car_number, owner_name) VALUES (?, ?)", (car_number, owner_name))
    conn.commit()

# Function to get a free parking slot
def get_free_slot():
    cursor.execute("SELECT slot_id FROM parking_slots WHERE occupied = 0 LIMIT 1")
    return cursor.fetchone()

# Function to allocate a parking slot
def allocate_slot(slot_id, car_number, owner_name):
    cursor.execute(
        "UPDATE parking_slots SET occupied = 1, car_number = ?, owner_name = ? WHERE slot_id = ?",
        (car_number, owner_name, slot_id)
    )
    conn.commit()

# Function to process a single frame
def process_frame(frame):
    # Convert frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform detection
    results = model.predict(rgb_frame, conf=0.5)  # Adjust confidence threshold as needed
    
    # Annotate the frame and extract plates
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    annotated_frame = frame.copy()
    plate_texts = []
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # Draw rectangle on the frame
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Crop the plate region
        plate_region = rgb_frame[y1:y2, x1:x2]
        # Use EasyOCR to extract text
        result = reader.readtext(plate_region, detail=0)
        plate_text = " ".join(result).strip()
        plate_texts.append(plate_text)
        
        # Check database for the plate number
        if plate_text and is_car_registered(plate_text):
            car_info = is_car_registered(plate_text)
            owner_name = car_info[2]  # Owner's name from database
            free_slot = get_free_slot()
            
            if free_slot:
                slot_id = free_slot[0]
                allocate_slot(slot_id, plate_text, owner_name)
                cv2.putText(annotated_frame, f"Access Granted: {plate_text} (Slot {slot_id})", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                st.success(f"Access granted! Slot {slot_id} allocated to {plate_text}.")
            else:
                cv2.putText(annotated_frame, f"No Free Slots: {plate_text}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                st.error("No free parking slots available!")
        else:
            cv2.putText(annotated_frame, f"Access Denied: {plate_text}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            st.error(f"Access denied for {plate_text}. Car not registered.")

    return annotated_frame, plate_texts

# Access live camera feed
def live_detection_page():
    # Initialize Streamlit session state
    if "detection_active" not in st.session_state:
        st.session_state.detection_active = False

    # Streamlit app interface
     # Custom title with adjusted size
    st.markdown(
        """
        <h1 style="font-size:30px; color:#007bff; text-align:left;">
            Live Number Plate Detection
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.write("This application detects number plates from your live camera feed.")
    
    # Create a placeholder for displaying the frame
    frame_placeholder = st.empty()

    # Buttons for controlling detection
    start_button = st.button(":blue[Start Detection]", key="start_button")
    stop_button = st.button(":blue[Stop Detection]", key="stop_button")

    # Handle button actions
    if start_button:
        st.session_state.detection_active = True
    if stop_button:
        st.session_state.detection_active = False

    cap = cv2.VideoCapture(0)  # 0 for default camera; replace with camera index if needed

    if not cap.isOpened():
        st.error("Error: Unable to access the camera.")
        return

    # Run detection loop
    while st.session_state.detection_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame. Exiting...")
            break
        
        # Process the current frame
        processed_frame, plate_texts = process_frame(frame)
        
        # Convert the processed frame to RGB format for Streamlit
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Update the frame in the placeholder
        frame_placeholder.image(
            rgb_frame, 
            caption="Live Number Plate Detection", 
            channels="RGB", 
            use_container_width=True
        )
        
        # Allow the app to respond to stop button during the loop
        if stop_button:
            st.session_state.detection_active = False
            break

    # Release the capture
    cap.release()

# Run the application
if __name__ == "__live_detection_page__":
    live_detection_page()
