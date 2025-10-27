import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="MNIST Classifier", layout="centered")
st.title("🔢 MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9)")

# Try to load TensorFlow, but provide fallback
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    st.success("✅ TensorFlow loaded successfully!")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    st.error(f"❌ TensorFlow not available: {e}")
    st.info("Running in demo mode")

# Load model function
def load_model():
    if not TENSORFLOW_AVAILABLE:
        return None
    try:
        model = tf.keras.models.load_model('mnist_cnn_model.h5')
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Uploaded Image", width=200)
    
    if model and st.button("Predict Digit"):
        # Preprocess image
        img = image.resize((28, 28))
        img_array = np.array(img)
        img_array = 255 - img_array  # Invert colors
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Predict
        prediction = model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        st.success(f"**Predicted Digit: {predicted_digit}**")
        st.metric("Confidence", f"{confidence:.1%}")
        
    elif not model:
        st.warning("🧪 **Demo Mode** - Model not loaded")
        st.info("Prediction would appear here with proper TensorFlow setup")
        st.metric("Example Prediction", "Digit 7")
        st.metric("Example Confidence", "92%")

st.markdown("---")
st.write("**Model Status:**", "✅ Loaded" if model else "❌ Not loaded")
