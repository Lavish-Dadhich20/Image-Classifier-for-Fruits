import os
import numpy as np
from PIL import Image

fruits = {
    'apple':  ([200, 50,  50],  [180, 30,  30]),
    'banana': ([240, 220, 50],  [220, 200, 30]),
    'orange': ([230, 130, 30],  [210, 110, 20]),
}

N_TRAIN = 80
N_VAL   = 20
IMG_SIZE = 100

for fruit, (base_color, dark_color) in fruits.items():
    for split, n in [('train', N_TRAIN), ('val', N_VAL)]:
        folder = os.path.join("dataset", split, fruit)
        os.makedirs(folder, exist_ok=True)

        for i in range(n):
            img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 240

            noise = np.random.randint(-15, 15, 3)
            color = np.clip(np.array(base_color) + noise, 0, 255)

            cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
            radius = np.random.randint(28, 38)

            for y in range(IMG_SIZE):
                for x in range(IMG_SIZE):
                    if (x - cx)**2 + (y - cy)**2 <= radius**2:
                        img[y, x] = color

            Image.fromarray(img).save(os.path.join(folder, f"{fruit}_{i}.jpg"))

print("Dataset created under ./dataset/")
print("Structure: dataset/train/{apple,banana,orange}/  and  dataset/val/{...}/")
