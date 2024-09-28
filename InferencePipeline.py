import numpy as np

def prepare_image(pil_image_obj):
    img = pil_image_obj.resize((150, 150))
    img = img.convert('L')
    img = np.array(img) / 255.0
    img = img.reshape((150, 150, 1))
    return img

def infer(model_obj, img):
    img = img[np.newaxis, ...]  # Add batch dimension (1, 150, 150, 1)
    logits = model_obj.predict(img)  # Predict
    pred = np.argmax(logits)

    lookup = {
        0:'buildings',
        1:'forest',
        2:'glacier',
        3:'mountain',
        4:'sea',
        5:'street'
    }

    return lookup[pred]
