import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# 📌 Step 1: Dataset Path
train_dir = "dataset"

# 📌 Step 2: Data Preprocessing (NO validation)
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=2,
    class_mode='binary'
)

# 📌 Step 3: Build CNN Model (Updated clean version)
model = models.Sequential([
    layers.Input(shape=(150,150,3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

# 📌 Step 4: Compile Model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 📌 Step 5: Train Model
history = model.fit(
    train_data,
    epochs=10
)

# 📌 Step 6: Save Model
model.save("cat_dog_model.h5")

print("✅ Model trained successfully!")