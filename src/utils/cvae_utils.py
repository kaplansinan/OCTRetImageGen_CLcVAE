import numpy as np
import pandas as pd
import os
import glob 
import random
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import cv2

import sys
sys.path.append('./')
from src.dataloader.contrastive_learning_loader import _denorm


def normalize_np_embedding_vector(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Load the numpy files
def load_np_embedding_feature(feature_path):
    feature = np.load(feature_path).astype(np.float16)
    feature = normalize_np_embedding_vector(feature)
    return feature #feature.reshape(-1,128)

# normalize between -1 and 1
def normalize_negative_one(img):
    normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    return 2*normalized_input - 1


def open_gray(fn):
    img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def read_image_3_channel(path):
    img = open_gray(path)
    img = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
#     img = img/255.
     # Apply model-specific preprocessing function
    img = normalize_negative_one(img)
    return  np.float32(img)


