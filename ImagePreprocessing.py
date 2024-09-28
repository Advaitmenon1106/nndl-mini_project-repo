import os
import numpy as np
from PIL import Image

def preprocess_data():
    X = []
    y = []
    y_encoded = []

    lookup = {
        'buildings':0,
        'forest':1,
        'glacier':2,
        'mountain':3,
        'sea':4,
        'street':5
    }

    root = './seg_train'
    for d in os.listdir(root):
        for img in os.listdir(f'{root}/{d}'):
            img_data = Image.open(f'{root}/{d}/{img}').convert('L')
            img_data = img_data.resize((150, 150))
            img_data = np.array(img_data)
            X.append(img_data)
            y.append(d)
    
    for val in y:
        y_encoded.append(lookup[val])
    
    return X, y_encoded

# _, y = preprocess_data()
# print(y)