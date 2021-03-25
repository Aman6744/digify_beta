import re
import cv2
import html
import string
import math
import numpy as np
from config import config

def adjust_to_see(img):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    img = cv2.warpAffine(img, M, (nW + 1, nH + 1))
    img = cv2.warpAffine(img.transpose(), M, (nW, nH))

    return img


def imread(img_path, target_size, predict):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)

    if predict:
        img = preprocess_image_predict(img)

    u, i = np.unique(img.flatten(), return_inverse=True)
    background_intensity = int(u[np.argmax(np.bincount(i))])
    return img, background_intensity

def preprocess_image(image_path, target_size, predict=False, augmentation=False):
    image, bg_intensity = imread(image_path, target_size, predict)
    (t_w, t_h, ch) = target_size
    (h, w) = image.shape
    fx = w/t_w
    fy = h/t_h
    f = max(fx, fy)
    newsize = (max(min(t_w, int(w / f)), 1), max(min(t_h, int(h / f)), 1))
    image = cv2.resize(image, newsize)
    (h, w) = image.shape
    background = np.ones((t_h, t_w), dtype=np.uint8) * bg_intensity
    row_freedom = background.shape[0]-image.shape[0]
    col_freedom = background.shape[1]-image.shape[1]
    row_off=0
    col_off=0
    if augmentation:
        if row_freedom:
            row_off = np.random.randint(0, row_freedom)
        if col_freedom:
            col_off = np.random.randint(0, col_freedom)
    else:
        row_off, col_off = row_freedom//2 , col_freedom//2
   
    background[row_off:row_off+h, col_off:col_off+w] = image
   
    image = cv2.transpose(background)
    return image

def augmentation(image_batch, 
        rotation_range=0, 
        scale_range=0, 
        height_shift_range=0, 
        width_shift_range=0,
        dilate_range=1, 
        erode_range=1):

    imgs = np.asarray(image_batch)
    _, h, w = imgs.shape

    background_intensity = []
    for img in imgs: 
        u, i = np.unique(img.flatten(), return_inverse=True)
        background_intensity.append(int(u[np.argmax(np.bincount(i))]))

    imgs = imgs.astype(np.float32)

    dilate_kernel = np.ones((int(np.random.uniform(1, dilate_range)),), np.uint8)
    erode_kernel = np.ones((int(np.random.uniform(1, erode_range)),), np.uint8)
    height_shift = np.random.uniform(-height_shift_range, height_shift_range)
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - scale_range, 1)
    width_shift = np.random.uniform(-width_shift_range, width_shift_range)
    trans_map = np.float32([[1, 0, width_shift * w], [0, 1, height_shift * h]])
    rot_map = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)

    trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
    rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
    affine_mat = rot_map_aff.dot(trans_map_aff)[:2, :]

    for i in range(_):
        imgs[i] = cv2.warpAffine(imgs[i], affine_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=background_intensity[i])
        imgs[i] = cv2.erode(imgs[i], erode_kernel, iterations=1)
        imgs[i] = cv2.dilate(imgs[i], dilate_kernel, iterations=1)
        if np.random.random() < 0.1:
        	imgs[i] = 255 - imgs[i]

    return imgs

def normalization(image_batch):
    imgs = np.asarray(image_batch).astype(np.float32)
    imgs = np.expand_dims(imgs / 255, axis=-1)
    return imgs

def preprocess_label(text, maxTextLength):
    std_text = text_standardize(text)
    strip_punc = std_text.strip(string.punctuation).strip()
    no_punc = std_text.translate(str.maketrans("", "", string.punctuation)).strip()

    length_valid = (len(std_text) > 1) and (len(std_text) < maxTextLength)
    text_valid = (len(strip_punc) > 1) or (len(no_punc) > 1)

    if (not length_valid) or (not text_valid):
        return (False, text)
    
    cost = 0
    for i in range(len(text)):
        if i != 0 and text[i] == text[i-1]:
            cost += 2
        else:
            cost += 1

        if cost > maxTextLength:
            return (False, text[:i])

    return (True, text)
    
RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(
    chr(768), chr(769), chr(832), chr(833), chr(2387),
    chr(5151), chr(5152), chr(65344), chr(8242)), re.UNICODE)
RE_RESERVED_CHAR_FILTER = re.compile(r'[¶¤«»]', re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}]'.format(re.escape(string.punctuation)), re.UNICODE)

LEFT_PUNCTUATION_FILTER = """!%&),.:;<=>?@\\]^_`|}~"""
RIGHT_PUNCTUATION_FILTER = """"(/<=>@[\\^_`{|~"""
NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE)


def text_standardize(text):
    if text is None:
        return ""

    text = html.unescape(text).replace("\\n", "").replace("\\t", "")

    text = RE_RESERVED_CHAR_FILTER.sub("", text)
    text = RE_DASH_FILTER.sub("-", text)
    text = RE_APOSTROPHE_FILTER.sub("'", text)
    text = RE_LEFT_PARENTH_FILTER.sub("(", text)
    text = RE_RIGHT_PARENTH_FILTER.sub(")", text)
    text = RE_BASIC_CLEANER.sub("", text)

    text = text.lstrip(LEFT_PUNCTUATION_FILTER)
    text = text.rstrip(RIGHT_PUNCTUATION_FILTER)
    text = text.translate(str.maketrans({c: f" {c} " for c in string.punctuation}))
    text = NORMALIZE_WHITESPACE_REGEX.sub(" ", text.strip())

    return text

def preprocess_image_predict(img):
    threshold_values = {}
    h = [1]

    def Hist(img):
       row, col = img.shape 
       y = np.zeros(256)
       for i in range(0,row):
          for j in range(0,col):
             y[img[i,j]] += 1
       return y


    def regenerate_img(img, threshold):
        row, col = img.shape 
        y = np.zeros((row, col))
        for i in range(0,row):
            for j in range(0,col):
                if img[i,j] > threshold:
                    y[i,j] = img[i,j]
                else:
                    y[i,j] = (-1*(threshold*(np.power((threshold-img[i,j])/threshold,config.gamma))-threshold)).astype(np.uint8) 
        return y



    def countPixel(h):
        cnt = 0
        for i in range(0, len(h)):
            if h[i]>0:
               cnt += h[i]
        return cnt


    def weight(s, e):
        w = 0
        for i in range(s, e):
            w += h[i]
        return w


    def mean(s, e):
        m = 0
        w = weight(s, e)
        for i in range(s, e):
            m += h[i] * i

        return m/float(w)


    def variance(s, e):
        v = 0
        m = mean(s, e)
        w = weight(s, e)
        for i in range(s, e):
            v += ((i - m) **2) * h[i]
        v /= w
        return v


    def threshold(h):
        cnt = countPixel(h)
        for i in range(1, len(h)):
            vb = variance(0, i)
            wb = weight(0, i) / float(cnt)
            mb = mean(0, i)

            vf = variance(i, len(h))
            wf = weight(i, len(h)) / float(cnt)
            mf = mean(i, len(h))

            V2w = wb * (vb) + wf * (vf)
            V2b = wb * wf * (mb - mf)**2

            if not math.isnan(V2w):
                threshold_values[i] = V2w


    def get_optimal_threshold():
        min_V2w = min(threshold_values.values())
        optimal_threshold = [k for k, v in threshold_values.items() if v == min_V2w]
        return optimal_threshold[0]

    h = Hist(img)
    threshold(h)
    op_thres = get_optimal_threshold()

    res = regenerate_img(img, op_thres)
    return res


if __name__ == "__main__":
    img1 = cv2.imread("t.png", 0)
    import matplotlib.pyplot as plt

    plt.subplot(121)
    plt.imshow(img1, cmap='gray')
    img2 = preprocess_image("t.png", (256, 64, 1), True)
    img2 = augmentation([img2],rotation_range=20.0, 
                        scale_range=0.05, 
                        height_shift_range=0.025, 
                        width_shift_range=0.05, 
                        erode_range=5, 
                        dilate_range=3)
    img2 = normalization(img2)[0]
    plt.subplot(122)
    plt.imshow(adjust_to_see(img2), cmap='gray')
    plt.show()
