from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

# ✅ Match file name here
model = load_model("cat_dog_model.h5")

img_path = "test.jpg"   # your test image

img = image.load_img(img_path, target_size=(150,150))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")