import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
from datetime import datetime

# ==== Theme Toggle ====
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

def set_theme(dark_mode):
    if dark_mode:
        st.markdown(
            """
            <style>
            [data-testid="stAppViewContainer"],
            [data-testid="stSidebar"],
            [data-testid="stAppViewBlockContainer"],
            [data-testid="stHeader"],
            [data-testid="stToolbar"] {
                background-color: #0e1117 !important;
                color: #fafafa !important;
            }
            .stButton>button {
                background-color: #21262d !important;
                color: #fafafa !important;
                border: none !important;
                border-radius: 6px !important;
            }
            input, textarea {
                background-color: #21262d !important;
                color: #fafafa !important;
                border: 1px solid #444c56 !important;
                border-radius: 4px !important;
            }
            div[data-testid="stFileUploaderDropzone"] {
                background-color: #21262d !important;
                border: 1px dashed #444c56 !important;
                color: #fafafa !important;
            }
            /* Force labels (like Username, Password, Dark Mode) to stay black */
            label, [data-testid="stCheckbox"] div p {
                color: black !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            [data-testid="stAppViewContainer"],
            [data-testid="stSidebar"],
            [data-testid="stAppViewBlockContainer"],
            [data-testid="stHeader"],
            [data-testid="stToolbar"] {
                background-color: white !important;
                color: black !important;
            }
            .stButton>button {
                background-color: #e0e0e0 !important;
                color: black !important;
                border-radius: 6px !important;
            }
            input, textarea {
                background-color: white !important;
                color: black !important;
                border: 1px solid #ccc !important;
                border-radius: 4px !important;
            }
            label, [data-testid="stCheckbox"] div p {
                color: black !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

# Dark mode toggle in sidebar
with st.sidebar:
    dark_mode_toggle = st.checkbox("🌙 Dark Mode", value=st.session_state["dark_mode"])
    st.session_state["dark_mode"] = dark_mode_toggle

set_theme(st.session_state["dark_mode"])

# ==== Authentication (Single Login Page) ====
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔐 Login")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    login_btn = st.button("Login")

    if login_btn:
        if username and password:  # Simple check
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            st.success(f"Welcome, {username}!")
            st.experimental_rerun()  # Refresh to load main app
        else:
            st.error("Please enter username and password.")
    st.stop()  # Stop here until logged in

# ==== Main App ====
st.title("Malaria Detection")

@st.cache_resource
def load_model():
    return YOLO("C:/Users/hp/best.pt")

model = load_model()

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01)

uploaded_file = st.file_uploader(
    "Upload blood smear image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    if image.size != (1024, 1024):
        image = image.resize((1024, 1024))

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image)

    with st.spinner("Analyzing blood sample..."):
        results = model.predict(img_array, conf=confidence, iou=iou)
        annotated_img = results[0].plot()

    st.image(annotated_img, caption="Detection Results", use_container_width=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/{st.session_state['user']}/"
    os.makedirs(save_dir, exist_ok=True)

    original_path = os.path.join(save_dir, f"input_{timestamp}.jpg")
    annotated_path = os.path.join(save_dir, f"output_{timestamp}.jpg")

    image.save(original_path)
    Image.fromarray(annotated_img).save(annotated_path)

    st.success(f"✅ Results saved to `{save_dir}`")

else:
    st.info("Upload a blood smear image to begin detection.")

if st.sidebar.button("🚪 Logout"):
    st.session_state["authenticated"] = False
    st.session_state.pop("user", None)
    st.experimental_rerun()
