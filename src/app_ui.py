import streamlit as st
import requests

st.title ("Plant Seedlings Classifier")

uploaded_file = st.file_uploader(
    "Select seed photo", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Selected photo", use_column_width=True)

    if st.button("Classify"):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        with st.spinner("Processing..."):
            response = requests.post("http://localhost:8000/predict", files=files)

        if response.status_code == 200:
            data = response.json()
            st.success(f"Class: {data['class']}")
            st.write(f"Confidence: {data['confidence']:.2f}")
        else:
            st.error("Something went wrong, try again.")