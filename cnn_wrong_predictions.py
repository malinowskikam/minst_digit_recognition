from cv2.cv2 import imwrite,resize, INTER_NEAREST

from cnn import load_model
from data import get_training_data_4d
import numpy as np

model = load_model()
data = get_training_data_4d()

predictions = np.argmax(model.predict(data[1]), axis=1)
classes = np.argmax(data[3], axis=1)
i=0

for image, true_class, predicted_class in zip(data[1],classes,predictions):
    if true_class != predicted_class:
        print(f"class {true_class} predicted as class {predicted_class}")
        img = (image.reshape(28, 28))*-255+255
        name = f"data\\wrongly_classified\\{i}_{true_class}as{predicted_class}.png"
        imwrite(name, resize(img,(280,280),interpolation=INTER_NEAREST))
        i += 1
