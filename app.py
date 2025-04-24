import streamlit as st
from model import DeepfakeDetector
from PIL import Image

def main():
    st.set_page_config(page_title="Ethical Deepfake Defence System", layout="centered")

    st.title("Ethical Deepfake Defence System")
    st.markdown("### Upload an image or video to check if it is Real or Deepfake.")

    # Sidebar with instructions
    st.sidebar.header("Instructions")
    st.sidebar.markdown(
        """
        1. Click on 'Browse files' or drag and drop an image or video file (jpg, jpeg, png, mp4, avi).
        2. Wait for the model to analyze the file.
        3. View the prediction result below the uploaded file.
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed for Ethical Deepfake Defence Project")

    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi"])
    if uploaded_file is not None:
        file_type = uploaded_file.type
        detector = DeepfakeDetector()

        if file_type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            label, confidence = detector.predict(image)
        elif file_type.startswith("video"):
            st.video(uploaded_file)
            label, confidence = detector.predict_video(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return

        # Display prediction with colored badge
        if label.lower() == "real":
            st.success(f"Prediction: **{label}** (Confidence: {confidence:.2f})")
        else:
            st.error(f"Prediction: **{label}** (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()
        