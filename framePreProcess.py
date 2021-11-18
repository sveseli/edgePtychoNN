from skimage.transform import resize
import numpy as np 

def frame_preprocess(frame, uid):
    resize_frame=(frame[2:-2,2:-2].reshape(128,4,128,4).mean(3).mean(1).astype('float32'))
    return resize_frame, uid

