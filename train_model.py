import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

# Data Generators
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    "./data/train",
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    "./data/validation",
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)

# Model Definition
base_model = InceptionV3(input_shape=(299, 299, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile and Train
model.compile(optimizer=RMSprop(learning_rate=0.00001),
              loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    verbose=2
)

model.save("model.h5")
