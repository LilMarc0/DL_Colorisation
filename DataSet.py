import numpy as np
import cv2 as cv
import os
import pdb

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image, ImageStat

def detect_color_image(file, thumb_size=40, MSE_cutoff=22, adjust_color_bias=True):
    pil_img = Image.open(file)
    bands = pil_img.getbands()
    try:
        if bands == ('R','G','B') or bands== ('R','G','B','A'):
            thumb = pil_img.resize((thumb_size,thumb_size))
            SSE, bias = 0, [0,0,0]
            if adjust_color_bias:
                bias = ImageStat.Stat(thumb).mean[:3]
                bias = [b - sum(bias)/3 for b in bias ]
            for pixel in thumb.getdata():
                mu = sum(pixel)/3
                SSE += sum((pixel[i] - mu - bias[i])*(pixel[i] - mu - bias[i]) for i in [0,1,2])
            MSE = float(SSE)/(thumb_size*thumb_size)
            if MSE <= MSE_cutoff:
                return False
            else:
                return True
    except:
        return False

def get_image(file, input_size, base_dir):
    # citim imaginea
    if detect_color_image(os.path.join(base_dir, file)):
        bgr_image = cv.imread(os.path.join(base_dir, file))
        # redimensionam imaginea conform parametrului self.network_input_size.
        bgr_image = cv.resize(bgr_image, input_size, interpolation=cv.INTER_AREA)
        # convertim imaginea in reprezentarea Lab.
        lab_image = cv.cvtColor(np.float32(bgr_image) / 255, cv.COLOR_BGR2Lab)
        # luam canalul L.
        gray_image, a_channel, b_channel = cv.split(lab_image)
        # luam canalale ab si le impartim la 128.
        a_channel = np.array(a_channel) / 128
        b_channel = np.array(b_channel) / 128
        gray_image = np.expand_dims(gray_image, axis=2)
        gt_image = cv.merge((a_channel, b_channel))
        return gray_image, bgr_image, gt_image
    else:
        return None

class DataSet:

    def __init__(self):
        self.training_dir = '/tmp/ramdisk/Dataset'
        self.test_dir = './test/'
        self.network_input_size = (256, 256)  # dimensiunea imaginilor de antrenare
        self.dir_output_images = './output'
        if not os.path.exists(self.dir_output_images):
            os.makedirs(self.dir_output_images)

        self.input_training_images,  self.ground_truth_training_images, self.ground_truth_bgr_training_images =\
            self.read_images(self.training_dir)
        self.input_test_images, self.ground_truth_test_images, self.ground_truth_bgr_test_images =\
            self.read_images(self.test_dir)


    def read_images(self, base_dir):
        files = os.listdir(base_dir)
        in_images = []  # imaginile de input, canalul L din reprezentarea Lab.
        gt_images = []  # imaginile de output (ground-truth), canalele ab din reprezentarea Lab.
        bgr_images = []  # imaginile in format BGR.

        partial = 20000
        with ProcessPoolExecutor(28) as exec:
            for res in tqdm(as_completed([exec.submit(get_image, file, self.network_input_size, base_dir) for file in files[:partial]]), total=partial):
                r = res.result()
                if r is not None:
                    gray_image, bgr_image, gt_image = r
                    in_images.append(gray_image)
                    bgr_images.append(bgr_image)
                    gt_images.append(gt_image)
                else:
                    continue
        print("FINAL DATASET: ", len(in_images))
        return np.array(in_images, np.float32), np.array(gt_images, np.float32), np.array(bgr_images, np.float32)

