import streamlit as st
from utils.ocr_utils import extract_text_from_image
from utils.classifier_utils import predict_document_type
from PIL import Image

st.title("📄 Document Type Identifier")

uploaded_file = st.file_uploader("Upload a document image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Extracting text..."):
        text = extract_text_from_image(image)
    
    st.subheader("Extracted Text")
    st.text(text[:500])

    if text.strip():
        prediction = predict_document_type(text)
        st.subheader("Predicted Document Type")
        st.success(prediction)
    else:
        st.error("No readable text found. Please upload a clearer image.")
