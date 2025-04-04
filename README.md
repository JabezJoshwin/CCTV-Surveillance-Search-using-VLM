# Visual Search on CCTV Footage using VLMs

This project is developed as part of the **Intel Unnati Internship Program**. It enables users to perform visual searches on CCTV footage using Vision-Language Models (VLMs), specifically OpenAI's CLIP model.

## Features
- Upload CCTV footage videos.
- Enter a text query to search for matching frames.
- Uses OpenAI's CLIP model to extract visual features.
- Implements FAISS for efficient similarity search.
- Displays top matching frames with confidence scores.

## Tech Stack
- **Python**
- **Streamlit** (for the user interface)
- **OpenAI CLIP** (for vision-language feature extraction)
- **FAISS** (for efficient similarity search)
- **OpenCV & PIL** (for video frame processing)

## Installation
```sh
pip install streamlit torch torchvision torchaudio openai-clip faiss-cpu opencv-python pillow numpy
```

## Usage
```sh
streamlit run app.py
```

## Demo
[![Watch the video](https://img.youtube.com/vi/r-1HDPAdEXk/maxresdefault.jpg)](https://www.youtube.com/watch?v=r-1HDPAdEXk)

## How It Works
1. The user uploads a CCTV footage video.
2. The system extracts features from video frames using CLIP.
3. A text query is entered, which is converted into a feature representation.
4. FAISS is used to find similar frames based on feature similarity.
5. The top matching frames are displayed with confidence scores.

## Project Structure
```
.
├── app.py  # Main application file
├── requirements.txt  # Dependencies
├── README.md  # Project documentation
```

## License
This project is for educational purposes under the **Intel Unnati Internship Program**.

---
For any queries, feel free to reach out!

