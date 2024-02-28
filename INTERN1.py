import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(weights='imagenet')

def detect_deepfake(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    
    # Top prediction
    label = decoded_predictions[0][1]
    confidence = decoded_predictions[0][2]
    
    return label, confidence

if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"
    label, confidence = detect_deepfake(image_path)
    print("Label:", label)
    print("Confidence:", confidence)
