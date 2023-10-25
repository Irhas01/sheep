import cv2
import os
import numpy as np
from sklearn import svm
from skimage.feature import hog
from sklearn.model_selection import train_test_split

# Data Preparation
data_directory = 'D:/TIF/Semester 5/Project Peternakan Kambing/sheep/images'

# Create empty lists to store features and labels
X = []  # Features
y = []  # Labels (1 for sheep, 0 for not sheep)

# Load and preprocess images from the dataset
for filename in os.listdir(data_directory):
    if filename.endswith(".jpg"):
        image_path = os.path.join(data_directory, filename)
        img = cv2.imread(image_path)

        # Preprocess the image (resize, convert to grayscale)
        img = cv2.resize(img, (64, 64))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract features (HOG, you can use other methods as well)
        features = hog(gray_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

        # Append features and labels
        X.append(features)
        if "sheep" in filename:
            y.append(1)
        else:
            y.append(0)

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Support Vector Machine (SVM) classifier
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Function to identify sheep in a frame
def identify_sheep(frame):
    # Preprocess the frame (resize, convert to grayscale)
    frame = cv2.resize(frame, (64, 64))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Extract features (HOG, you can use other methods as well)
    features = hog(gray_frame, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    
    # Predict whether the frame contains a sheep or not
    prediction = clf.predict([features])
    
    return prediction

# Weight Estimation Phase with Regression
def estimate_weight(length_mm, breadth_mm):
    # Define regression equation parameters
    c = 0.064443
    d = 0.010059

    # Calculate the weight
    weight = c * breadth_mm + d * length_mm

    return weight

# Pre-processing Phase
def preprocess_image(frame):
    # Crop out legs and neck (Update coordinates as needed)
    cropped_frame = frame[50:400, 50:450]  # Example coordinates

    # Adjust brightness (You can experiment with different methods)
    gamma = 1.5
    adjusted_frame = np.power(cropped_frame / 255.0, gamma) * 255.0
    adjusted_frame = adjusted_frame.astype(np.uint8)  # Ensure it's in the correct format

    # Convert to grayscale
    gray = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2GRAY)

    # Threshold to create a binary image (Adjust threshold value)
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return binary_image

# Segmentation Phase (K-Means Clustering)
def kmeans_segmentation(binary_image):
    # Define K-Means parameters
    k = 2  # Number of clusters (You may need to adjust this)
    max_iterations = 100
    accuracy = 0.2

    # Flatten the binary image to create a 1D array
    pixels = binary_image.reshape((-1, 1))

    # Convert to float32 data type
    pixels = np.float32(pixels)

    # Define K-Means criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, accuracy)

    # Apply K-Means clustering
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape the labeled image to the original shape
    segmented_image = labels.reshape(binary_image.shape)

    return segmented_image

# Open the camera
cap = cv2.VideoCapture(0)  # Adjust camera index as needed

while True:
    ret, frame = cap.read()
    
    # Identify sheep in the frame
    is_sheep = identify_sheep(frame)
    
    if is_sheep:
        # Pre-processing
        preprocessed_frame = preprocess_image(frame)

        # Segmentation
        segmented_frame = kmeans_segmentation(preprocessed_frame)

        # Convert segmented_frame to CV_8UC1 data type
        segmented_frame = np.uint8(segmented_frame)

        # Find contours of connected components
        contours, _ = cv2.findContours(segmented_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (assuming it's the sheep)
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate the dimensions (length and breadth) of the sheep
            x, y, w, h = cv2.boundingRect(largest_contour)
            length_mm = w
            breadth_mm = h

            # Estimate weight
            estimated_weight = estimate_weight(length_mm, breadth_mm)

            # Display a message or draw a bounding box around the sheep
            cv2.putText(frame, "Sheep", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Weight: {estimated_weight:.2f} kg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Sheep Identification', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()