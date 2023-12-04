#MODEL LIBS
import os
import cv2
import numpy as np
import base64
import tensorflow as tf
import random
import warnings
# Ignore deprecated warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from PIL import Image
from keras.optimizers import RMSprop
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Lambda
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K  # Import Keras backend
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from django.core.files.uploadedfile import InMemoryUploadedFile
import matplotlib.pyplot as plt
import seaborn as sns
from django.core.files import File
from django.db.models.fields.files import FieldFile

# Define the get_random_image function
def get_random_image(img_size):
    # Create a random image with values between 0 and 1
    random_image = np.random.rand(img_size[0], img_size[1], 3)
    return random_image

img_width, img_height = 300, 150
input_shape = (img_width, img_height, 1)

# Load data
dataset_path = "C:\\Documents\\THESIS\\DATASETS\\SIGNATURE"
original_path = os.path.join(dataset_path, "ORIGINAL_SIGNATURES")
forged_path = os.path.join(dataset_path, "FORGED_SIGNATURES")

# List of signature names
signature_names = ["Nepomuceno ", "Jamion ", "C. Vasquez ", "Dignadice ", "Panolino ", "Mangubat ", 
                   "Arzaga ","Toledo ", "Delcoro ", "Relatado ", "Timosa ", "Bacaser ", "Realubit ", 
                   "Obaredes ", "Banzuelo ", "Galicia ", "Tejada ", "Abia ", "Cu ", "Padul ", 
                   "Marquez ", "Suranol ", "Salonoy ", "Badilla ", "Obanana ", "Pasamonte ", 
                   "Tabangay ", "Villono ", "Recarze ", "S. Vasquez "]  

images = []
labels = []

for class_index, signature_name in enumerate(signature_names):
    original_class_path = os.path.join(original_path)
    forged_class_path = os.path.join(forged_path)

    # Load genuine images
    for stroke_number in range(1, 26):
        # Load original image
        original_filename = f"{signature_name}{stroke_number}.jpg"
        original_img_path = os.path.join(original_class_path, original_filename)

        # print(f"Loading original image: {original_img_path}")

        original_img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)

        if original_img is None:
            # print(f"Error loading original image: {original_img_path}")
            pass
        else:
            original_img = cv2.resize(original_img, (img_width, img_height))
            original_img = original_img.reshape((img_width, img_height, 1))

        images.extend([original_img])
        labels.extend([class_index])  # 0 for original

    # Load forged images
    for stroke_number in range(1, 21):  # Adjusted to go up to stroke_number 20
        # Load forged image
        forged_filename = f"{signature_name}{stroke_number}.jpg"
        forged_img_path = os.path.join(forged_class_path, forged_filename)

        # print(f"Loading forged image: {forged_img_path}")

        forged_img = cv2.imread(forged_img_path, cv2.IMREAD_GRAYSCALE)

        if forged_img is None:
            # print(f"Error loading forged image: {forged_img_path}")
            pass
        else:
            forged_img = cv2.resize(forged_img, (img_width, img_height))
            forged_img = forged_img.reshape((img_width, img_height, 1))

        images.extend([forged_img])
        labels.extend([class_index])  # 1 for forged

# Check and ensure all images have the same dimensions
image_shapes = [img.shape if img is not None else None for img in images]
# print("Image shapes:", image_shapes)

if not all(shape == (img_width, img_height, 1) or shape is None for shape in image_shapes):
    raise ValueError("Not all images have the same dimensions.")

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# print(f'{"="*50} labels got {"="*50}')
# print(f'labels: {labels}')
# print(f'{"="*100}')

# Reshape the images to have 3 dimensions (assuming grayscale images)
images = images.reshape((-1, img_width, img_height, 1))

def create_siamese_model(input_shape=input_shape):
    model = Sequential()
    model.add(Convolution2D(16, (8, 8), strides=(1, 1), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(32, (4, 4), strides=(1, 1), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(64, (2, 2), strides=(1, 1), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification, so 1 neuron with sigmoid activation
    return model

# Insert the new code here
siamese_model = create_siamese_model(input_shape)
rms = RMSprop()
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, restore_best_weights=True)
callback_early_stop_reduceLROnPlateau = [earlyStopping]

siamese_model.compile(optimizer=rms, loss='binary_crossentropy', metrics=["accuracy"])
siamese_model.summary()

batch_size = 32

# Generate additional images
one = []
zero = []

img_size = (300, 150)

for x in range(200):
    img = get_random_image(img_size)

    a, b = random.randrange(0, img_size[0] // 4), random.randrange(0, img_size[0] // 4)
    c, d = random.randrange(img_size[0] // 2, img_size[0]), random.randrange(img_size[0] // 2, img_size[0])

    value = random.sample([True, False], 1)[0]
    if value == False:
        img[a:c, b:d, 0] = 25
        img[a:c, b:d, 1] = 25
        img[a:c, b:d, 2] = 25
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).convert('L')) / 255
        one.append(img)
    else:
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).convert('L')) / 255
        zero.append(img)

# Convert additional images to numpy arrays and reshape
additional_zero = np.array(zero).reshape(-1, img_width, img_height, 1)
additional_one = np.array(one).reshape(-1, img_width, img_height, 1)

# Concatenate images with additional images
images = np.concatenate([images, additional_one, additional_zero])
labels = np.concatenate([labels, np.ones(len(additional_one)), np.zeros(len(additional_zero))])

# Concatenate images with additional images
images = np.concatenate([images, additional_one, additional_zero])
labels = np.concatenate([labels, np.ones(len(additional_one)), np.zeros(len(additional_zero))])

shuffled_indices = np.random.permutation(len(images))
images = images[shuffled_indices]
labels = labels[shuffled_indices]

labels = labels.astype(int)  # Convert labels to integers

# Insert the new code here
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

total_sample_size = 50
test_sample_size = 200
dim1, dim2 = 300, 150

x_pair = np.zeros([total_sample_size, 2, dim1, dim2, 1])
y = np.zeros([total_sample_size, 1])

x_pair_test = np.zeros([test_sample_size, 2, dim1, dim2, 1])
y_test = np.zeros([test_sample_size, 1])

# Initialize lists to store filenames during testing
tested_genuine_filenames = []
tested_forged_filenames = []

for x in range(total_sample_size):
    value = random.sample([True, False], 1)[0]
    if value:
        pair = random.choices(one, k=2)
        x_pair[x, 0, :, :, 0] = pair[0]
        x_pair[x, 1, :, :, 0] = pair[1]
        y[x] = 1
    else:
        x_pair[x, 0, :, :, 0] = random.choices(one, k=1)[0]
        x_pair[x, 1, :, :, 0] = random.choices(zero, k=1)[0]
        y[x] = 0

for x in range(test_sample_size):
    value = random.sample([True, False], 1)[0]
    if value:
        pair = random.choices(one, k=2)
        x_pair_test[x, 0, :, :, 0] = pair[0]
        x_pair_test[x, 1, :, :, 0] = pair[1]
        y_test[x] = 1
        # Store filenames
        tested_genuine_filenames.append(pair[0])
        tested_genuine_filenames.append(pair[1])
    else:
        x_pair_test[x, 0, :, :, 0] = random.choices(one, k=1)[0]
        x_pair_test[x, 1, :, :, 0] = random.choices(zero, k=1)[0]
        y_test[x] = 0
        # Store filenames
        tested_genuine_filenames.append(pair[0])  # Assuming you want the genuine filename
        tested_forged_filenames.append(x_pair_test[x, 1, :, :, 0])  # Assuming you want the forged filename

model2 = Model(inputs=siamese_model.input, outputs=siamese_model.layers[-2].output)

input_dim = (dim1, dim2, 1)

img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)

feat_vecs_a = model2(img_a)
feat_vecs_b = model2(img_b)

# Create a new Lambda layer using the defined function
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)
callback_early_stop_reduceLROnPlateau = [early_stopping]

model = Model(inputs=[img_a, img_b], outputs=distance)
model.compile(loss=contrastive_loss, optimizer=adam_optimizer, metrics=[accuracy])
model.summary()

model.fit([x_pair[:, 0], x_pair[:, 1]], y, validation_data=([x_pair_test[:, 0], x_pair_test[:, 1]], y_test), batch_size=batch_size, verbose=1, epochs=10, callbacks=callback_early_stop_reduceLROnPlateau)

model.save_weights('model_weights.h5')
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())
print('saved')

distances = model.predict([x_pair_test[:, 0], x_pair_test[:, 1]])

threshold = 0.5

binary_predictions = distances < threshold

svm_features = distances.flatten()

svm_threshold = 0.1
svm_labels = (svm_features < svm_threshold).astype(int)

svm_features = svm_features.reshape(-1, 1)

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    svm_features, svm_labels, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_svm, y_train_svm)

svm_predictions = svm_model.predict(X_test_svm)

svm_accuracy = accuracy_score(y_test_svm, svm_predictions)
print("SVM Accuracy: {:.2%}".format(svm_accuracy))
print("Classification Report:\n", classification_report(y_test_svm, svm_predictions))

true_positives = np.sum(np.logical_and(binary_predictions == 1, y_test == 1))
false_positives = np.sum(np.logical_and(binary_predictions == 1, y_test == 0))
true_negatives = np.sum(np.logical_and(binary_predictions == 0, y_test == 0))
false_negatives = np.sum(np.logical_and(binary_predictions == 0, y_test == 1))

FRR = false_negatives / (true_positives + false_negatives)
FAR = false_positives / (false_positives + true_negatives)
ACC = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)

print(f"Threshold: {threshold}")
print("False Rejection Rate (FRR): {:.2%}".format(FRR))
print("False Acceptance Rate (FAR): {:.2%}".format(FAR))
print("Accuracy Rate (ACC): {:.2%}".format(ACC))
print("="*50)

# Load Siamese model and create a feature extraction function
siamese_model = create_siamese_model(input_shape)  # Load your Siamese model
feature_extraction_model = Model(inputs=siamese_model.input, outputs=siamese_model.layers[-2].output)

def extract_features_siamese(img):
    img = cv2.resize(img, (img_width, img_height))
    img = img.reshape((1, img_width, img_height, 1))
    features = feature_extraction_model.predict(img)
    return features

# svm model is already trained and defined globally
def predict(img_file, signature_file):
    # Process img_file
    if isinstance(img_file, InMemoryUploadedFile):
        img_array = np.frombuffer(img_file.open().read(), np.uint8)
        if img_array.size == 0:
            raise ValueError("Empty image data")
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    elif isinstance(img_file, File):  # Assuming File is imported from django.core.files
        img = cv2.imread(img_file.path)
    else:
        raise ValueError("Unsupported image file type")

    # Process signature_file
    if isinstance(signature_file, FieldFile):
        signature_array = np.frombuffer(signature_file.read(), np.uint8)
        if signature_array.size == 0:
            raise ValueError("Empty signature data")
        signature = cv2.imdecode(signature_array, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("Unsupported signature file type")
    
    # Extract features using Siamese model
    img_features_siamese = extract_features_siamese(img)
    signature_features_siamese = extract_features_siamese(signature)

    # Compute distance between images using Siamese features
    distance_siamese = np.linalg.norm(img_features_siamese - signature_features_siamese)

    # Assuming distance is computed as in your code
    distance_min = 0  # Minimum possible distance
    distance_max = 150  # Maximum possible distance (you need to set this based on your problem)

    # Invert the distance
    inverted_distance = distance_max - distance_siamese

    # Scale the inverted distance to [0-100%]
    scaled_distance = 100 * inverted_distance / (distance_max - distance_min)

    # Predict using SVM model for Siamese features
    svm_prediction_siamese = svm_model.predict([[distance_siamese]])

    print(f'{"="*50} svm_prediction_siamese got {"="*50}')
    print(f'img_features_siamese: {img_features_siamese}, len: {len(img_features_siamese)}, type: {type(img_features_siamese)}')
    print(f'signature_features_siamese: {signature_features_siamese}, len: {len(signature_features_siamese)}, type: {type(signature_features_siamese)}')
    print(f'distance_siamese: {distance_siamese}')
    print(f'Scaled Inverted Distance: {scaled_distance}%')

    print(f'svm_prediction_siamese: {svm_prediction_siamese}')
    print(f'{"="*100}')

    # Return the percentage distance based on Siamese features
    return scaled_distance