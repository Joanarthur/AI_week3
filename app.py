import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set up the page
st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")

# Title and description
st.title("🎯 MNIST Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9) and see the AI model predict it!")

# Load your trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('mnist_cnn_model.h5')
        return model
    except:
        st.error("Model file not found. Please ensure 'mnist_cnn_model.h5' is in the same directory.")
        return None

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption="Uploaded Image", width=200)
    
    if model is not None:
        if st.button("🔍 Predict Digit"):
            # Preprocess the image
            img = image.resize((28, 28))
            img_array = np.array(img)
            img_array = 255 - img_array  # Invert colors (MNIST has white on black)
            img_array = img_array.astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            
            # Make prediction
            prediction = model.predict(img_array, verbose=0)
            predicted_digit = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # Display results
            st.success(f"**Predicted Digit: {predicted_digit}**")
            st.metric("Confidence", f"{confidence:.1%}")
            
            # Show confidence bars for all digits
            st.write("**Confidence for all digits:**")
            for i, conf in enumerate(prediction[0]):
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    st.write(f"**{i}:**")
                with col_b:
                    st.progress(float(conf))
                    st.write(f"{conf:.1%}")

# Model information
st.markdown("---")
st.subheader("📊 Model Information")
st.write("""
- **Model Type**: Convolutional Neural Network (CNN)
- **Training Data**: MNIST dataset (60,000 images)
- **Framework**: TensorFlow/Keras
- **Test Accuracy**: >98%
""")

st.info("💡 **Tip**: For best results, use clear images of centered digits on a dark background.")
