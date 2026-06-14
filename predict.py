import numpy as np
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


IMG_SIZE = 100
CLASSES  = ['apple', 'banana', 'orange']   # update if your dataset has different/more classes

model = load_model("fruit_classifier.h5")

img_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpeg"

img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
x   = image.img_to_array(img) / 255.0
x   = np.expand_dims(x, axis=0)

preds = model.predict(x)[0]
# ensure preds is a 1-D array
preds = np.asarray(preds).ravel()
idx = int(np.argmax(preds))

# If model output size doesn't match CLASSES, warn and print safely
if preds.shape[0] != len(CLASSES):
    print(f"Warning: model returned {preds.shape[0]} outputs but CLASSES has {len(CLASSES)} labels.")

pred_label = CLASSES[idx] if idx < len(CLASSES) else f"class_{idx}"
print(f"Prediction : {pred_label}")
print(f"Confidence : {preds[idx]*100:.1f}%")

for i in range(preds.shape[0]):
    label = CLASSES[i] if i < len(CLASSES) else f"class_{i}"
    print(f"  {label:10s} {preds[i]*100:.1f}%")
