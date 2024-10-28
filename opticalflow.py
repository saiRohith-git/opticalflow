import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Constants and Parameters
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
EPOCHS = 10

# Load DataFrames
train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")

# Helper Functions
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

def load_video(video_path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(0)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(weights="imagenet", include_top=False, pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

label_processor = tf.keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_df["tag"]))

def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values.tolist()

    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for idx, path in enumerate(video_paths):
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
        temp_frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

        video_length = frames.shape[1]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            temp_frame_features[0, j, :] = feature_extractor.predict(frames[:, j, :, :, :])
        temp_frame_mask[0, :length] = 1

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels

# Prepare the training data
(train_data, train_masks), train_labels = prepare_all_videos(train_df, "./train")

# Convert labels to numerical values
train_labels = np.array(label_processor(train_labels))

# Create the sequence model
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    x = keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01))(frame_features_input, mask=mask_input)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LSTM(32, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LSTM(16, kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", 
        optimizer="adam",
        metrics=["accuracy"]
    )
    return rnn_model

sequence_model = get_sequence_model()

# Train the model
sequence_model.fit([train_data, train_masks], train_labels, epochs=EPOCHS)

# Save the model weights
sequence_model.save_weights("./video_classification_project/video_classifier.weights.h5")

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    video_length = frames.shape[1]
    length = min(MAX_SEQ_LENGTH, video_length)
    for j in range(length):
        frame_features[0, j, :] = feature_extractor.predict(frames[:, j, :, :, :])
    frame_mask[0, :length] = 1

    return frame_features, frame_mask

def live_action_detection():
    class_vocab = label_processor.get_vocabulary()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        frames = []
        for _ in range(MAX_SEQ_LENGTH):
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            frames.append(frame)
        
        if len(frames) == MAX_SEQ_LENGTH:
            frames = load_video(frames)
            frame_features, frame_mask = prepare_single_video(frames)
            probabilities = sequence_model.predict([frame_features, frame_mask])[0]
                
            prediction = class_vocab[np.argmax(probabilities)]
            confidence = np.max(probabilities) * 100
            print(f"Prediction: {prediction}, Confidence: {confidence:.2f}%")

            cv2.putText(frame, f'{prediction} ({confidence:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Live Action Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

live_action_detection()
