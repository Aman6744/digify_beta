import matplotlib.pyplot as plt
import cv2
import numpy as np
from data.tokenizer import Tokenizer
from data.generator import DataGenerator_tf
from config import config
from data import data_preprocessor as pp

tokenizer = Tokenizer(charset=config.charset)
dg = DataGenerator_tf("train")
ds = dg.create_dataset()
for i, l in ds.take(1):
    print(i.shape, tokenizer.sequences_to_texts(np.swapaxes([l.numpy()], 0, 1)))
    plt.subplot(121)
    plt.imshow(pp.adjust_to_see(i[0].numpy()), cmap="gray")
    plt.subplot(122)
    plt.imshow(pp.adjust_to_see(i[1].numpy()), cmap="gray")
    plt.show()