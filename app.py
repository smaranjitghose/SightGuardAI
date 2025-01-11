import os
import tempfile
import cv2
from io import BytesIO
from pathlib import Path

import streamlit as st
import moondream as md
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load environment variables
load_dotenv()

# Cache the Moondream API client
@st.cache_resource
def load_moondream_api(api_key):
    try:
        return md.vl(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Moondream API: {str(e)}")
        return None

# Function to extract frames from video and their timestamps
def extract_frames_with_timestamps(video_path, interval=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    success, image = cap.read()
    count = 0

    while success:
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_sec = timestamp_ms / 1000.0

        if count % (interval * frame_rate) == 0:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(img)
            timestamps.append(timestamp_sec)

        success, image = cap.read()
        count += 1

    cap.release()  # Release the video capture object
    print(f"Total frames captured: {len(frames)}")
    return frames, timestamps

# Function to calculate cosine similarity between two descriptions
def calculate_similarity(prev_description, current_description):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([prev_description, current_description])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def main():
    st.set_page_config(
        page_title="SightGuardAI: Automatic Surveillance Tagging",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("👁️ SightGuardAI")
    st.write("Upload a surveillance video to extract frames, generate descriptions, and identify key frames.")

    # Add API key input in sidebar
    with st.sidebar:
        api_key = st.text_input("Enter your Moondream API Key", type="password")
        if api_key:
            os.environ["MOONDREAM_API_KEY"] = api_key

        uploaded_file = st.file_uploader(
            "Upload a surveillance video",
            type=["mp4", "avi", "mov"]
        )

    # Initialize Moondream API client if API key is provided
    if api_key:
        model = load_moondream_api(api_key)
        if model is None:
            st.error("Failed to initialize Moondream API. Please check your API key.")
            return
    else:
        st.warning("Please enter your Moondream API Key in the sidebar.")
        return

    if uploaded_file:
        # Save uploaded video to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
        temp_file.close()  # Close the file to release the handle

        # Display the uploaded video
        st.video(uploaded_file)

        # Add an "Analyze" button
        if st.button("Analyze"):
            with st.spinner("Analyzing video..."):
                # Extract frames and timestamps
                frames, timestamps = extract_frames_with_timestamps(video_path, interval=1)  # Extract 1 frame per second

                # Process each frame using the Moondream API
                descriptions = []
                key_frames = []
                similarity_threshold = 0.8  # Adjust this threshold as needed

                prev_description = ""
                for i, frame in enumerate(frames):
                    try:
                        encoded_image = model.encode_image(frame)
                        description = model.caption(encoded_image)["caption"]

                        # Calculate similarity with the previous frame's description
                        if prev_description:
                            similarity = calculate_similarity(prev_description, description)
                            if similarity < similarity_threshold:  # Key frame if similarity is below threshold
                                key_frames.append((timestamps[i], frame))

                        descriptions.append((timestamps[i], description))
                        prev_description = description
                    except Exception as e:
                        st.error(f"Error processing frame {i + 1}: {str(e)}")
                        continue

                # Create a DataFrame for frames and descriptions
                frame_data = {
                    "Frame": [f"Frame {i + 1}" for i in range(len(frames))],
                    "Timestamp (s)": [f"{timestamp:.2f}" for timestamp in timestamps],
                    "Description": [description for _, description in descriptions],
                    "Key Frame": ["Yes" if (timestamp, frame) in key_frames else "No" for timestamp, frame in zip(timestamps, frames)]
                }
                df = pd.DataFrame(frame_data)

                # Display the table
                st.header("Frame Descriptions and Key Frames")
                st.dataframe(df, use_container_width=True)

                # Display key frames in a grid layout
                if key_frames:
                    st.header("Key Frames")
                    num_columns_key_frames = 3  # Number of columns for key frames grid
                    num_rows_key_frames = (len(key_frames) + num_columns_key_frames - 1) // num_columns_key_frames  # Calculate number of rows needed

                    for row in range(num_rows_key_frames):
                        cols = st.columns(num_columns_key_frames)
                        for col in range(num_columns_key_frames):
                            index = row * num_columns_key_frames + col
                            if index < len(key_frames):
                                timestamp, frame = key_frames[index]
                                cols[col].image(frame, caption=f"Key Frame {index + 1} at {timestamp:.2f}s")

        # Clean up temporary file
        try:
            os.unlink(video_path)
        except Exception as e:
            st.error(f"Could not delete temporary file: {e}")

if __name__ == "__main__":
    main()