import subprocess
import sys
st.write(subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True).stdout)
import streamlit as st
import sqlite3
import pandas as pd
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import easyocr
from home import home
from streamlit_option_menu import option_menu
from live_plate_detection import live_detection_page

# Connect to the SQLite database
conn = sqlite3.connect('car_parking.db', check_same_thread=False)
cursor = conn.cursor()

# Create tables with owner_name field in parking_slots
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
);
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS parking_slots (
    slot_id INTEGER PRIMARY KEY,
    occupied INTEGER DEFAULT 0,
    car_number TEXT DEFAULT NULL,
    owner_name TEXT DEFAULT NULL
);
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS registered_cars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    car_number TEXT UNIQUE,
    owner_name TEXT
);
""")
conn.commit()

# Load YOLOv8 pre-trained model
model = YOLO("best.pt")  # Replace with your YOLOv8 model

# Utility functions
def add_user(username, password):
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()

def authenticate_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    return cursor.fetchone()

def get_free_slot():
    cursor.execute("SELECT slot_id FROM parking_slots WHERE occupied = 0 LIMIT 1")
    return cursor.fetchone()

def allocate_slot(slot_id, car_number, owner_name):
    cursor.execute("UPDATE parking_slots SET occupied = 1, car_number = ?, owner_name = ? WHERE slot_id = ?", 
                   (car_number, owner_name, slot_id))
    conn.commit()

def free_slot(slot_id):
    cursor.execute("UPDATE parking_slots SET occupied = 0, car_number = NULL, owner_name = NULL WHERE slot_id = ?", 
                   (slot_id,))
    conn.commit()

def register_car(car_number, owner_name):
    cursor.execute("INSERT INTO registered_cars (car_number, owner_name) VALUES (?, ?)", (car_number, owner_name))
    conn.commit()

def is_car_registered(car_number):
    cursor.execute("SELECT * FROM registered_cars WHERE car_number = ?", (car_number,))
    return cursor.fetchone()

def detect_number_plate(image):
    results = model(image)
    detections = results[0].boxes.xyxy.cpu().numpy()
    return detections, results[0].boxes.conf.cpu().numpy()

def extract_text_from_region(image, bbox):
    x_min, y_min, x_max, y_max = map(int, bbox)
    cropped_image = image[y_min:y_max, x_min:x_max]
    reader = easyocr.Reader(['en'])
    text = reader.readtext(cropped_image, detail=0)
    return text[0] if text else None

# Streamlit Pages
def signup_page():
    st.title(":blue[Signup]")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button(":blue[Signup]"):
        try:
            add_user(username, password)
            st.success("User registered successfully!")
        except:
            st.error("Username already exists!")

def login_page():
    st.title(":blue[Login]")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button(":blue[Login]"):
        user = authenticate_user(username, password)
        if user:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state['onboarding_page'] = 0  # Start onboarding
            st.success("Login successful!")
        else:
            st.error("Invalid username or password!")
    st.markdown("<br>", unsafe_allow_html=True)
    st.write(":blue['Login'] if you are registered. If not, navigate to :blue['SignUp'] to get registered.")


def number_plate_detection_page():
    # Custom title with adjusted size
    st.markdown(
        """
        <h1 style="font-size:30px; color:#007bff; text-align:left;">
            Vehicle Number Plate Detection
        </h1>
        """,
        unsafe_allow_html=True,
    )

    # File uploader for detecting number plates
    uploaded_file = st.file_uploader("Upload an image of the car", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Detect number plates using YOLOv8
        detections, confidences = detect_number_plate(image_np)

        if detections.size > 0:
            for bbox, confidence in zip(detections, confidences):
                car_number = extract_text_from_region(image_np, bbox)
                if car_number:
                    st.write(f"Detected Number: {car_number} (Confidence: {confidence:.2f})")
                    car_info = is_car_registered(car_number)
                    if car_info:
                        owner_name = car_info[2]
                        free_slot = get_free_slot()
                        if free_slot:
                            allocate_slot(free_slot[0], car_number, owner_name)
                            st.success(f"Access granted! Slot allocated: {free_slot[0]} for {owner_name}")
                        else:
                            st.error("No free parking slots available.")
                    else:
                        st.error("Access denied! Car not registered.")

                        # Add an expander to register the car
                        with st.expander("Register this car (Detected Number)") :
                            st.write("Enter the owner's name to register this car.")
                            owner_name = st.text_input(f"Owner Name for {car_number}", key=f"owner_{car_number}")
                            if st.button("Register Car (Detected Number)", key=f"register_{car_number}") :
                                try:
                                    register_car(car_number, owner_name)
                                    st.success(f"Car {car_number} registered to {owner_name}!")
                                except sqlite3.IntegrityError:
                                    st.error("Car number already exists in the database.")
                else:
                    st.error("Unable to extract text from the detected number plate.")
        else:
            st.error("No number plates detected in the image.")

def parking_slot_management_page():
    # Custom title with adjusted size
    st.markdown(
        """
        <h1 style="font-size:30px; color:#007bff; text-align:left;">
            Parking Slots Management
        </h1>
        """,
        unsafe_allow_html=True,
    )
    
    # Add new parking slots
    num_slots = st.number_input("Enter the number of parking slots to add:", min_value=1, step=1)
    if st.button(":blue[Add Slots]"):
        for i in range(num_slots):
            cursor.execute("INSERT OR IGNORE INTO parking_slots (slot_id) VALUES (?)", (i+1,))
        conn.commit()
        st.success(f"{num_slots} slots added successfully!")
    
    # Show current parking slots
    st.markdown(
        """
        <h1 style="font-size:30px; color:#007bff; text-align:left;">
            Current Parking Slots Status
        </h1>
        """,
        unsafe_allow_html=True,
    )
    slots_df = pd.read_sql_query("SELECT * FROM parking_slots", conn)
    st.dataframe(slots_df)

    # Create closer columns for buttons
    col1, col2, col3, = st.columns([1,1,1])

    with col2:
        if st.button(":blue[Refresh Table]"):
            slots_df = pd.read_sql_query("SELECT * FROM parking_slots", conn)

    with col1:
        if st.button(":blue[Free All Parking Slots]"):
            cursor.execute("UPDATE parking_slots SET occupied = 0, car_number = NULL, owner_name = NULL")
            conn.commit()
            st.success("All parking slots have been freed and reset.")

    # Remove a parking slot entry by owner name
    st.markdown(
        """
        <h1 style="font-size:30px; color:#007bff; text-align:left;">
            Free Parking Slot By Owner Name
        </h1>
        """,
        unsafe_allow_html=True,
    )
    owner_name_to_remove = st.text_input("Enter the owner's name to free the parking slot")
    if st.button(":blue[FreeUpSlot]"):
        if owner_name_to_remove:
            cursor.execute("UPDATE parking_slots SET occupied = 0, car_number = NULL, owner_name = NULL WHERE owner_name = ?", (owner_name_to_remove,))
            conn.commit()
            st.success(f"Parking slot freed for owner {owner_name_to_remove}.")
        else:
            st.error("Please provide an owner's name.")

def manage_registered_cars_page():
     # Custom title with adjusted size
    st.markdown(
        """
        <h1 style="font-size:30px; color:#007bff; text-align:left;">
            Vehicle Registering and Management
        </h1>
        """,
        unsafe_allow_html=True,
    )
    
# Add new plate section
    with st.expander("Manually Register a Car"):
        st.write("Enter the car number and owner's name to register the car manually.")
        manual_car_number = st.text_input("Car Number", key="manual_car_number")
        manual_owner_name = st.text_input("Owner Name", key="manual_owner_name")
        if st.button("Register Car (Manual)", key="register_manual"):
            if manual_car_number and manual_owner_name:
                try:
                    register_car(manual_car_number, manual_owner_name)
                    st.success(f"Car {manual_car_number} registered to {manual_owner_name}!")
                except sqlite3.IntegrityError:
                    st.error("Car number already exists in the database.")
            else:
                st.error("Please provide both the car number and owner's name.")

    # Show all registered cars
    cars_df = pd.read_sql_query("SELECT * FROM registered_cars", conn)
    st.write("### Registered Cars")
    st.dataframe(cars_df)

# Button to refresh the table records after an update or delete
    if st.button(":blue[Refresh Table]"):
        cars_df = pd.read_sql_query("SELECT * FROM registered_cars", conn)

    # Delete a car by owner name
     # Custom title with adjusted size
    st.markdown(
        """
        <h1 style="font-size:30px; color:#007bff; text-align:left;">
            Delete Vehicle
        </h1>
        """,
        unsafe_allow_html=True,
    )
    owner_name_to_delete = st.text_input("Enter the owner's name of the car to delete")
    if st.button(":blue[Delete Car]"):
        if owner_name_to_delete:
            cursor.execute("DELETE FROM registered_cars WHERE owner_name = ?", (owner_name_to_delete,))
            conn.commit()
            st.success(f"Car with owner {owner_name_to_delete} deleted successfully!")
        else:
            st.error("Please provide an owner's name.")

    # Update car information
     # Custom title with adjusted size
    st.markdown(
        """
        <h1 style="font-size:30px; color:#007bff; text-align:left;">
            Update Vehicle Information
        </h1>
        """,
        unsafe_allow_html=True,
    )
    car_to_update = st.text_input("Enter the car number to update")
    new_owner_name = st.text_input("Enter the new owner's name")
    if st.button(":blue[Update]"):
        if car_to_update and new_owner_name:
            cursor.execute("UPDATE registered_cars SET owner_name = ? WHERE car_number = ?", 
                           (new_owner_name, car_to_update))
            conn.commit()
            st.success(f"Car {car_to_update}'s owner updated to {new_owner_name}!")
        else:
            st.error("Please provide both the car number and new owner's name.")

# Streamlit onboarding pages
def onboarding_page():
    st.title(":blue[Welcome to the Vehicle Recognition and Parking Management System]")

    # Define onboarding content with titles
    onboarding_content = [
        {"title": ":violet[Vehicle Registration]", "img": "piclumen-1733510906990.png", 
         "text": "Welcome! This system will help you in the detection and recognition of vehicles through number plate detection from an image and live camera. Based on registered vehicles in the system, it will check whether the vehicle is valid or not. If valid\n:blue[Access granted] otherwise :blue[Access denied]."},
        {"title": ":violet[Parking Slot Management]", "img": "piclumen-1733512501144.jpg", 
         "text": "This system also manages real-time parking and automatic parking slot allocation. If a free parking slot is available it will allocate the slot to the vehicle if not, it shows a message :blue['There is no more space in the parking']."},
        {"title": ":violet[Registering Vehicle]", "img": "piclumen-1733513643881.png", 
         "text": "The system offers easy scalability with manual and automatic registration features allowing users to effortlessly register vehicles with owner names from the dashboard."},
        {"title": ":violet[Features Summary]", "img": "piclumen-1733566403303.jpg", 
         "text": "This system includes the following functions and features:\n\n1. Vehicle Detection and Access Management\n2. Real-Time Parking and Slot Allocation\n3. Scalability and Registration Features\n\nNow, let's enter the main application!"},
    ]

    # Initialize the onboarding page index
    if "onboarding_page" not in st.session_state:
        st.session_state.onboarding_page = 0

    current_page = st.session_state.onboarding_page

    # Display page title, image, and text
    st.subheader(onboarding_content[current_page]["title"])
    col1, col2 = st.columns(2)
    with col1:
        st.image(onboarding_content[current_page]["img"], use_container_width=True)
    with col2:
        st.write(onboarding_content[current_page]["text"])

    # Navigation buttons
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button(":blue[Skip]"):
            st.session_state.onboarding_page = -1  # End onboarding
    with col2:
        if st.button(":blue[Next]"):
            if current_page < len(onboarding_content) - 1:
                st.session_state.onboarding_page += 1
            else:
                st.session_state.onboarding_page = -1  # End onboarding


# Main logic
# Initialize session state variables if not already done
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'onboarding_page' not in st.session_state:
    st.session_state['onboarding_page'] = -1

# Main application
if st.session_state['logged_in']:
    if st.session_state.onboarding_page >= 0:
        onboarding_page()
    else:
        # Sidebar menu using option_menu with compact style
        st.sidebar.title(f"Welcome {st.session_state['username']}")
        page = option_menu(
            "Main Menu",
            ["Home", "Image Detection", "Live Detection", "Registered Vehicles", "Parking Slots"],
            icons=["house", "camera", "camera-reels", "car-front", "geo"],
            menu_icon="list",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "3px", "background-color": "#f8f9fa"},
                "icon": {"color": "blue", "font-size": "14px"},
                "menu-title": {"font-size": "16px", "font-weight": "bold", "color": "#333333"},
                "nav-link": {
                    "font-size": "11px",
                    "text-align": "left",
                    "margin": "2px",
                    "padding": "5px",
                },
                "nav-link-selected": {"background-color": "#007bff"},
            },
        )

        # Navigate to the selected page
        if page == "Home":
            home()
        elif page == "Image Detection":
            number_plate_detection_page()
        elif page == "Live Detection":
            live_detection_page()
        elif page == "Registered Vehicles":
            manage_registered_cars_page()
        elif page == "Parking Slots":
            parking_slot_management_page()
else:
    # Sidebar menu for login and signup with compact style
    page = option_menu(
        'User_Profile Menu',
        ["Login", "Signup"],
        icons=["box-arrow-in-right", "person-plus"],
        menu_icon="list",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "3px", "background-color": "#f8f9fa"},
            "icon": {"color": "blue", "font-size": "14px"},
            "menu-title": {"font-size": "16px", "font-weight": "bold", "color": "black"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "2px",
                "padding": "5px",
            },
            "nav-link-selected": {"background-color": "#007bff"},
        },
    )

    # Navigate to the selected page
    if page == "Login":
        login_page()
    elif page == "Signup":
        signup_page()

