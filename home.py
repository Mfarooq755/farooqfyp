import streamlit as st

def home():
    # Custom title with adjusted size
    st.markdown(
        """
        <h1 style="font-size:30px; color:#007bff; text-align:left;">
            Vehicle Recognition and Parking Management System
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.write("Welcome to the smart parking management system!")
    #st.markdown("<br>", unsafe_allow_html=True)
    st.write("Navigate through the tabs to perform actions:")

    # Buttons for navigation
    st.success("For Image Detection Goto 'image detection'")
    st.warning("For Live Detection Goto 'live detection'")
    st.info("For Registering Vehicle Management Goto 'registered vehicles'")
    st.error("For Parking Slots Management Goto 'parking slots")
