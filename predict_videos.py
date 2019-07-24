import os
import numpy as np

from PIL import Image

from keras.models import load_model
from keras_preprocessing import image

v_frames_dn = 'data/072319_02/cam1'
v_frames_files = os.listdir(v_frames_dn)

classifier = load_model('trained_models/video_models/072319_01/classifier_072319_01.h5')

frame_predictions = {}

for frame_fn in v_frames_files:
    frame_img = Image.open(os.path.join(v_frames_dn, frame_fn))

    frame_img = frame_img.resize((128, 128))

    frame_img_array = np.array(frame_img)


    # img_reloaded = Image.fromarray(frame_img_array)
    # frame_img.show()
    # img_reloaded.show()


    frame_img_array = np.expand_dims(frame_img_array, axis=0)

    frame_predictions[float(frame_fn.strip('.jpg'))] = classifier.predict(frame_img_array)

prediction_arrary = np.asarray(list(frame_predictions.items()))