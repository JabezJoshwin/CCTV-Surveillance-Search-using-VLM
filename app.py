import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import faiss
import cv2
import tempfile

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to extract features from an image
def extract_features(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.cpu().numpy()

# Calculate confidence score based on distances
def calculate_confidence(distances):
    max_distance = np.max(distances)
    min_distance = np.min(distances)
    confidence_scores = 1 - (distances - min_distance) / (max_distance - min_distance)
    return confidence_scores

# Streamlit UI
st.title("Visual Search on CCTV Footage using VLMs")
st.write("Upload a video and enter a text query to search for matching frames.")

# Upload video
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

# Enter text query
text_query = st.text_input("Enter a text query (e.g., 'a person wearing a red shirt')")

if video_file and text_query:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    # Load video
    cap = cv2.VideoCapture(tfile.name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Extract features from video frames
    frame_features = []
    frame_images = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        features = extract_features(frame_pil)
        frame_features.append(features)
        frame_images.append(frame_rgb)
    cap.release()

    # Convert features to numpy array and store in FAISS
    d = frame_features[0].shape[1] if frame_features else 512  # Automatically determine dimension
    index = faiss.IndexFlatL2(d)
    frame_features = np.vstack(frame_features)
    index.add(frame_features)

    # Encode text query
    text = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)

    # Search for similar frames
    query_features = text_features.cpu().numpy()
    D, I = index.search(query_features, k=5)  # k is the number of top matches

    # Calculate confidence scores
    confidence_scores = calculate_confidence(D[0])

    # Display results
    threshold = 0.5  # Set a threshold for confidence score
    if len(I) == 0 or len(I[0]) == 0 or np.all(confidence_scores < threshold):
        st.write("No matching frames found for query '{}' with sufficient confidence.".format(text_query))
    else:
        st.write("Top matching frames for query '{}':".format(text_query))
        for idx, confidence in zip(I[0], confidence_scores):
            if idx < len(frame_images) and confidence >= threshold:  # Minimal fix for out-of-range indexing
                st.image(frame_images[idx], caption="Frame {} (Confidence: {:.2f})".format(idx + 1, confidence), use_column_width=True)