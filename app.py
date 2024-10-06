import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(
    page_title="Potato Disease Classification"
)




@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('potato_model.h5')
    return model


with st.spinner('Model is being loaded..'):
    model = load_model()





def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])
if(app_mode=="Home"):
    st.header("**Potato Plant Disease Recognition System**")
    image_path = "dash.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    # Potato Disease Recognition System

## Overview
The **Potato Disease Recognition System** is a machine learning-based application designed to classify potato plant leaves into three categories:  
- **Potato___Early_blight**  
- **Potato___Late_blight**  
- **Potato___healthy**

This system helps farmers and agricultural experts quickly identify common potato diseases by analyzing images of potato leaves. The model used for classification leverages deep learning to ensure high accuracy and efficiency in detection.

## Features
- **Image Upload**: Users can upload images of potato leaves from their local device.
- **Disease Detection**: The system analyzes the uploaded image and classifies it into one of the three categories: Early Blight, Late Blight, or Healthy.
- **Confidence Score**: The system provides a confidence percentage for the predicted classification.
- **User-Friendly Interface**: Built with Streamlit, the application is easy to navigate, even for users with no technical background.
- **Real-Time Results**: The application delivers instant feedback upon image upload.

 ### How It Works
1. **Upload the Image**: Users can either upload an image file.
2. **Model Processing**: The system processes the image using a pre-trained deep learning model (`potato_model.h5`) to predict the class.
3. **Display Results**: The predicted class and the confidence score are displayed on the screen.

## Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/) (version 1.38.0)
- **Backend**: FastAPI (for future API integration)
- **Machine Learning**: TensorFlow 2.8.0 for model training and prediction
- **Model Type**: Image classification using a Convolutional Neural Network (CNN)
- **Supported Input Types**: Image upload (JPEG, PNG)
    """)
# About Page
elif (app_mode == "About"):
    st.header("About")
    st.markdown("""
    # About the Potato Disease Recognition System

## Purpose
The **Potato Disease Recognition System** was developed to help farmers and agricultural professionals quickly identify common diseases affecting potato plants. By leveraging modern machine learning techniques, this system aims to:
- **Improve Crop Management**: Early detection of diseases allows farmers to take preventive measures and minimize crop loss.
- **Increase Yield**: Timely interventions based on accurate disease identification can help maintain healthy crops, ultimately increasing yields.
- **Simplify Diagnosis**: The tool simplifies disease identification by removing the need for complex testing or manual inspection by experts.

## Key Features
- **Ease of Use**: The system has an intuitive interface built using Streamlit, allowing users to upload images from their local devices for analysis.
- **Accuracy**: The system leverages a deep learning model trained on a comprehensive dataset to classify potato leaves with a high degree of accuracy.
- **Confidence Scores**: Each prediction is accompanied by a confidence score to provide users with additional insights into the model's certainty.
- **Real-Time Results**: Users receive instant feedback upon uploading an image, allowing for timely decision-making.

## Why It Matters
Potatoes are a staple crop worldwide, and diseases like Early Blight and Late Blight pose a significant threat to their yield. By identifying these diseases early, farmers can reduce losses, apply targeted treatments, and contribute to sustainable farming practices.

## Technology Behind the System
The core technology behind the Potato Disease Recognition System includes:
- **Deep Learning Model**: A convolutional neural network (CNN) that classifies images into three categories: Early Blight, Late Blight, and Healthy.
- **Streamlit Interface**: A web-based interface that simplifies the interaction with the model, making it accessible even to non-technical users.
- **TensorFlow**: The backend framework used for model training and inference.

## Future Vision
This project is an ongoing effort, and we plan to introduce additional features such as:
- **Support for Other Crops**: Expanding the model to detect diseases in other crops.
- **API Integration**: Offering an API for external systems to integrate the disease detection capabilities.
- **Mobile Application**: A dedicated mobile app for even easier field access.

## Contributors
- **Priyansh Lunawat** - Lead Developer and Jr.Machine Learning Engineer


## Contact
If you have any questions, feedback, or would like to contribute to the project, please reach out at:
- **Email**: priyanshlunawat2207@gmail.com
 """)
elif (app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    file = st.file_uploader("", type=["jpg", "png"])
    if file is None:
        st.text("Please upload an image file")
    else:

        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, model)
        class_names = ['Early blight', 'Late blight', 'Healthy']
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100  # Confidence percentage

        result_string = f"Prediction: {predicted_class} with {confidence:.2f}% confidence"

        if predicted_class == 'Healthy':
            st.success(result_string)
        else:
            st.warning(result_string)

    html_link = """
        Made by <a href="https://www.linkedin.com/in/priyansh-lunawat-6a68b624b/" style="color:green;" target="_blank">Priyansh Lunawat</a>
        """
    st.markdown(html_link, unsafe_allow_html=True)


