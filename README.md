**Skin Disease Classification using Convolutional Neural Networks**
**Introduction**
This project focuses on developing a Convolutional Neural Network (CNN) model to classify different types of skin diseases from images. The model is trained to recognize and categorize images into one of 14 skin disease categories. With proper training and optimization, the model achieves an impressive accuracy of 93%, demonstrating its effectiveness in skin disease detection.

**Dataset**
The dataset consists of images of skin diseases, each labeled with one of the 14 disease types. The images are preprocessed and resized to 128x128 pixels to standardize the input for the CNN model.

**Project Structure**
model_training.ipynb: The main Jupyter Notebook containing the code for data loading, preprocessing, model architecture, training, and evaluation.
README.md: Project documentation.
images/: Directory containing sample images from the dataset (if applicable).
data/: Directory containing the dataset and CSV files (update with your actual data paths).
Prerequisites
Python 3.x
TensorFlow 2.x
Keras
scikit-learn
NumPy
Pandas
Matplotlib
Pillow (PIL)
OpenCV (cv2)

Getting Started
1. Clone the Repository
```
git clone https://github.com/your-username/skin-disease-classification.git
cd skin-disease-classification
```

2. Set Up the Environment
It's recommended to use a virtual environment:
```
python -m venv venv
# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install Required Packages
```
pip install -r requirements.txt

```
```
pip install tensorflow keras scikit-learn numpy pandas matplotlib pillow opencv-python
```

4. Data Preparation

- Dataset Location: Organize your dataset into training, validation, and test sets.
- CSV File: Ensure you have a CSV file (image_paths.csv) containing image paths and corresponding disease labels.
- Google Drive (Optional): If using Google Colab, mount your Google Drive to access the dataset stored there.

5. Running the Code
Open the Jupyter Notebook model_training.ipynb in Google Colab or your local environment and run the cells sequentially.

Key Steps:
- Mount Google Drive: To access the dataset stored in Google Drive (if applicable).
- Import Libraries: All necessary libraries are imported at the beginning.
- Data Loading: Images and labels are loaded using paths specified in the CSV file.
- Data Preprocessing:
Images are resized to 128x128 pixels.
Images are normalized by scaling pixel values to the range [0, 1].
Optional: Apply data augmentation techniques to enhance the dataset.
- Model Architecture:
A CNN model is defined using Keras Sequential API.
The architecture includes convolutional layers, max-pooling layers, batch normalization, dropout layers, and dense layers.
- Compilation:
The model is compiled with an optimizer (e.g., Adam), loss function (categorical cross-entropy), and evaluation metrics (e.g., accuracy).
- Training:
The model is trained on the training data with validation on the validation set.
Class weights are calculated to handle class imbalance.
Early stopping can be used to prevent overfitting.
- Evaluation:
After training, the model is evaluated on the validation/test set.
Accuracy and loss curves are plotted to visualize training progress.
- Results:
The model achieves an accuracy of 93% on the validation/test set.
Confusion matrices and classification reports can be generated for detailed performance analysis.
- Results
Accuracy: The model achieved an accuracy of 93% on the test set.
Loss: The training and validation loss decreased steadily over epochs, indicating good learning behavior.
- Plots:
Accuracy Plot: Shows the increase in accuracy over epochs for both training and validation sets.
Loss Plot: Shows the decrease in loss over epochs for both training and validation sets.
- Model Architecture
The CNN model architecture includes:

Convolutional Layers: Extract features from images using filters.
MaxPooling Layers: Reduce spatial dimensions to minimize computational complexity.
Batch Normalization: Normalize outputs to accelerate training and improve performance.
Dropout Layers: Prevent overfitting by randomly dropping units during training.
Flatten Layer: Converts the 2D feature maps into a 1D feature vector.
Dense Layers: Fully connected layers that perform classification based on extracted features.
Output Layer: Uses softmax activation to output probabilities for each of the 14 classes..
Model Summary:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 126, 126, 32)      896
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 63, 63, 32)        0
_________________________________________________________________
batch_normalization (BatchNo (None, 63, 63, 32)        128
_________________________________________________________________
dropout (Dropout)            (None, 63, 63, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 61, 61, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 30, 30, 64)        0
_________________________________________________________________
batch_normalization_1 (Batch (None, 30, 30, 64)        256
_________________________________________________________________
dropout_1 (Dropout)          (None, 30, 30, 64)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 128)       73856
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 14, 128)       0
_________________________________________________________________
batch_normalization_2 (Batch (None, 14, 14, 128)       512
_________________________________________________________________
dropout_2 (Dropout)          (None, 14, 14, 128)       0
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________
dense (Dense)                (None, 128)               3211392
_________________________________________________________________
batch_normalization_3 (Batch (None, 128)               512
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256
_________________________________________________________________
batch_normalization_4 (Batch (None, 64)                256
_________________________________________________________________
dropout_4 (Dropout)          (None, 64)                0
_________________________________________________________________
dense_2 (Dense)              (None, 32)                2080
_________________________________________________________________
batch_normalization_5 (Batch (None, 32)                128
_________________________________________________________________
dropout_5 (Dropout)          (None, 32)                0
_________________________________________________________________
dense_3 (Dense)              (None, 14)                462
=================================================================
Total params: 3,312,230
Trainable params: 3,311,366
Non-trainable params: 864
_________________________________________________________________
```
Load the Model:
```
from tensorflow.keras.models import load_model
model = load_model('path_to_saved_model.h5')
```
Prepare Input Image:

Load and preprocess the image in the same way as the training images (resize, normalize).
Make Predictions:
```
import numpy as np
predictions = model.predict(preprocessed_image)
predicted_class = np.argmax(predictions, axis=1
```
Interpret Results:

Map the predicted_class index to the actual disease label using the label encoding used during training.
Code Snippets
Mount Google Drive (If using Google Colab)
```
from google.colab import drive
drive.mount('/content/drive')
```
Import Libraries

```
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Conv2D, MaxPooling2D
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import os
Data Loading and Preprocessing
```
# Define paths
```
parent_folder = '/content/drive/MyDrive/ML H1 PROJECT/test'
csv_file = '/content/drive/MyDrive/image_paths.csv'

# Load the CSV file
attributes_df = pd.read_csv(csv_file)

# Initialize lists
images = []
disease_labels = []

# Create a mapping from disease type to integer
Disease_type_to_int = {name: idx for idx, name in enumerate(attributes_df['Disease_type'].unique())}

# Load images and labels
for index, row in attributes_df.iterrows():
    image_name = row['Image_Path']
    disease_type = row['Disease_type']
    image_path = os.path.join(parent_folder, image_name)
    if os.path.isfile(image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128), Image.LANCZOS)
        img = img.filter(ImageFilter.SHARPEN)
        img_array = np.array(img).astype('float32') / 255.0
        images.append(img_array)
        disease_labels.append(Disease_type_to_int[disease_type])

# Convert to NumPy arrays
images = np.array(images)
disease_labels = tf.keras.utils.to_categorical(np.array(disease_labels))
```
Model Training
```
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(images, disease_labels, test_size=0.2, random_state=30)

# Define the model architecture (as shown in Model Architecture section)
model = Sequential()
# ... [Add layers as per the architecture]

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,
)
```
Plotting Accuracy and Loss
```
# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```
Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss improvements or fixes.
