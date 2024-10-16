import streamlit as st
import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom attention layer (needed for loading the model)
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='glorot_uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# Function to load and filter dataset based on user selection
def load_test_dataset_by_user_selection(category_group, fabric_class):
    base_dir = '/content/Fabric FDD/Fabric_classification/Fabric classification'

    subcategories = {
        'Weaving': {'gray': ['stain', 'damage', 'broken thread', 'holes', 'Non defective'],
                    'dyed': ['stain', 'damage', 'broken thread', 'holes', 'Non defective'],
                    'printed': ['stain', 'damage', 'broken thread', 'holes', 'Non defective']},
        'Knitting': {'gray': ['stain', 'damage', 'broken thread', 'holes', 'Non defective'],
                     'dyed': ['stain', 'damage', 'broken thread', 'holes', 'Non defective'],
                     'printed': ['stain', 'damage', 'broken thread', 'holes', 'Non defective']}
    }

    if category_group not in subcategories or fabric_class not in subcategories[category_group]:
        raise ValueError("Invalid category or sub-category selection.")

    selected_subcategory = subcategories[category_group][fabric_class]

    images = []
    labels = []
    categories = list(selected_subcategory)
    category_counts = {category: 0 for category in categories}

    for label, defect in enumerate(categories):
        category_dir = os.path.join(base_dir, category_group, fabric_class, defect)
        if not os.path.exists(category_dir):
            st.write(f"Directory {category_dir} does not exist. Skipping this category.")
            continue
        for filename in os.listdir(category_dir):
            if filename.endswith('.jpg'):
                img = cv2.imread(os.path.join(category_dir, filename))
                img = cv2.resize(img, (128, 128))  # Resize image to 128x128
                images.append(img)
                labels.append(label)
                category_counts[defect] += 1

    images = np.array(images)
    labels = np.array(labels)

    st.write(f"Image counts per defect category for {category_group} - {fabric_class}: {category_counts}")

    return images, labels, categories


# Preprocess and evaluate the model based on user input
def test_saved_model_by_user_input(saved_model_path, category_group, fabric_class):
    # Load the dataset based on the user's selection
    images, labels, categories = load_test_dataset_by_user_selection(category_group, fabric_class)

    if len(images) == 0:
        st.error("No images found. Please check the dataset path and ensure images are available.")
        return

    # Preprocess the images (normalize pixel values)
    images = images.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    labels = to_categorical(labels, num_classes=len(categories))

    # Reshape images for LSTM input
    images = images.reshape(images.shape[0], 1, 128, 128, 3)

    # Load the pre-trained model
    model = load_model(saved_model_path, custom_objects={'AttentionLayer': AttentionLayer})

    # Evaluate the model on the test dataset
    loss, accuracy = model.evaluate(images, labels, verbose=0)
    st.write(f'Test Loss for {category_group} - {fabric_class}: {loss}')
    st.write(f'Test Accuracy for {category_group} - {fabric_class}: {accuracy}')

    # Get predictions for the test set
    predY = model.predict(images)
    predY_classes = np.argmax(predY, axis=1)
    trueY_classes = np.argmax(labels, axis=1)

    # Print classification report
    unique_true_labels = np.unique(trueY_classes)
    class_report = classification_report(
        trueY_classes, predY_classes,
        target_names=[categories[i] for i in unique_true_labels],
        output_dict=True,
        zero_division=0
    )
    st.write(f"Classification report for {category_group} - {fabric_class}:")
    st.write(class_report)

    # Print confusion matrix
    conf_matrix = confusion_matrix(trueY_classes, predY_classes)
    st.write(f"Confusion matrix for {category_group} - {fabric_class}:")
    st.write(conf_matrix)

    # Plot confusion matrix
    fig, ax = plt.subplots()
    ax.matshow(conf_matrix, cmap='Blues')
    plt.title(f'Confusion Matrix - {category_group} {fabric_class}')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(np.arange(len(categories)), categories, rotation=90)
    plt.yticks(np.arange(len(categories)), categories)
    st.pyplot(fig)


# Streamlit UI
st.title("Fabric Defect Detection and Classification")

# User selects the fabric type (Weaving or Knitting)
fabric_type = st.selectbox("Select fabric type", ["Weaving", "Knitting"])

# User selects the fabric class (Gray, Dyed, Printed)
fabric_class = st.selectbox("Select fabric class", ["gray", "dyed", "printed"])

# Upload the trained model file
model_file = st.file_uploader("Upload the trained model (.h5)", type=["h5"])

# If the user has uploaded a model, run the test
if model_file is not None:
    with open("saved_model.h5", "wb") as f:
        f.write(model_file.read())
    
    # Button to start testing the model on the selected dataset
    if st.button("Run Test"):
        try:
            test_saved_model_by_user_input("saved_model.h5", fabric_type, fabric_class)
        except Exception as e:
            st.error(f"Error: {str(e)}")
