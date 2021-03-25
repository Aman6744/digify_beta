import h5py
import string
import numpy as np
import tensorflow as tf

from data import data_preprocessor as pp
from data.tokenizer import Tokenizer
from config import config


class DataGenerator_tf():
    def __init__(self, partition):
        
        self.source_path = config.source_path
        self.partition = partition
        self.charset = config.charset
        self.maxTextLength = config.maxTextLength
        self.batch_size = config.batch_size
        self.buf_size = config.buf_size
        self.prefetch_size = config.prefetch_size
        self.tokenizer = Tokenizer()

        with h5py.File(self.source_path, "r") as f:
            self.imgs = f[self.partition]["image"][:]
            self.labels = f[self.partition]["label"][:]

        self.size = len(self.labels)

    def preprocessor_helper(self, x, y):
        y = y.numpy()
        x = x.numpy()

        if y.any():
            y_ = []
            for line in y:
                seq = self.tokenizer.texts_to_sequences(line.decode())[0]
                padded_seq = np.pad(seq, (0, self.maxTextLength-len(seq)))
                y_.append(padded_seq)

            y = np.array(y_)

        if self.partition in ["train"]:
            x = pp.augmentation(x, 
                    rotation_range=config.rotation_range, 
                    scale_range=config.scale_range, 
                    height_shift_range=config.height_shift_range, 
                    width_shift_range=config.width_shift_range, 
                    erode_range=config.erode_range, 
                    dilate_range=config.dilate_range)

        x = pp.normalization(x)

        if y.any():
            return x, y
        else:
            return x

    def get_img_label(self, x):
        index = x.numpy()
        if self.partition in ["test"]:
            return self.imgs[index]
        else:
            return self.imgs[index], self.labels[index]

    def create_dataset(self):
        indexes = [i for i in range(self.size)]
        if self.partition in ["train"]:
            np.random.shuffle(indexes)

        index_ds = tf.data.Dataset.from_tensor_slices(indexes)
        if self.partition in ["train"]:
            ds = index_ds.map(lambda x: tf.py_function(self.get_img_label, [x], [tf.uint8, tf.string])).shuffle(self.buf_size).batch(self.batch_size)
            final_ds = ds.map(lambda x,y: tf.py_function(self.preprocessor_helper, [x,y], [tf.float32, tf.float32]))
        elif self.partition in ["valid"]:
            ds = index_ds.map(lambda x: tf.py_function(self.get_img_label, [x], [tf.uint8, tf.string])).batch(self.batch_size)
            final_ds = ds.map(lambda x,y: tf.py_function(self.preprocessor_helper, [x,y], [tf.float32, tf.float32]))
        else:
            ds = index_ds.map(lambda x: tf.py_function(self.get_img_label, [x], [tf.uint8])).batch(self.batch_size)
            final_ds = ds.map(lambda x: tf.py_function(self.preprocessor_helper, [x,False], [tf.float32]))

        return final_ds.prefetch(self.prefetch_size)


class Datagenerator(tf.keras.utils.Sequence):
    def __init__(self, source_path, partition, charset, maxTextLength, batch_size=32, buf_size=1000):
        self.maxTextLength = maxTextLength
        self.tokenizer = Tokenizer(charset=charset)
        self.batch_size = batch_size
        self.partition = partition
        self.dataset = h5py.File(source_path, 'r')[self.partition]
        self.size = self.dataset['label'].shape[0]
        self.steps = int(np.ceil(self.size/self.batch_size))
        self.buf_size = buf_size
        
        if self.partition in ['train'] and self.buf_size:
            self.img_buf = self.dataset['image'][0:self.buf_size]
            self.lab_buf = self.dataset['label'][0:self.buf_size]

        # for p in self.partitions:
        #     self.size[p] = self.dataset[p]['image'].shape[0]
        #     self.steps[p] = int(np.ceil(self.size[p]/self.batch_size))
        #     self.index[p] = 0
    
    def __getitem__(self, idx):
        if self.partition in ['valid', 'test'] or not self.buf_size:
            index = idx*self.batch_size
            until = index+self.batch_size

            x = np.array(self.dataset['image'][index:until]) 
            if self.partition in ['train']:
                x = pp.augmentation(x, 
                        rotation_range=config.rotation_range, 
                        scale_range=config.scale_range, 
                        height_shift_range=config.height_shift_range, 
                        width_shift_range=config.width_shift_range, 
                        erode_range=config.erode_range, 
                        dilate_range=config.dilate_range)
            x = pp.normalization(x)
            if self.partition in ['valid', 'train']:
                y = self.dataset['label'][index:until]
                # y = [self.tokenizer.texts_to_sequences(word.decode())[0] for word in y]
                # y = np.array([np.pad(np.asarray(seq), (0, self.maxTextLength-len(seq)), constant_values=(-1, self.PAD)) for seq in y])
                y_ = []
                for line in y:
                    seq = self.tokenizer.texts_to_sequences(line.decode())[0]
                    padded_seq = np.pad(seq, (0, self.maxTextLength-len(seq)))
                    y_.append(padded_seq)

                y = np.array(y_)

                return (x, y)
            return x

        else :
            index = idx*self.batch_size + self.buf_size
            until = index+self.batch_size

            zipped = list(zip(self.img_buf, self.lab_buf))
            np.random.shuffle(zipped)

            X, Y = zip(*zipped)
            X = list(X)
            Y = list(Y)

            x = np.array(X[:self.batch_size])
            y = Y[:self.batch_size]

            if until < self.size:
                X[:self.batch_size] = self.dataset['image'][index:until]
                Y[:self.batch_size] = self.dataset['label'][index:until]

            elif index < self.size:
                X = X[until-self.size:]
                Y = Y[until-self.size:]
                until = self.size
                X[:until-index] = self.dataset['image'][index:until]
                Y[:until-index] = self.dataset['label'][index:until]

            else:
                X = X[self.batch_size:]
                Y = Y[self.batch_size:]

            self.img_buf = X
            self.lab_buf = Y

            x = pp.augmentation(x, 
                    rotation_range=config.rotation_range, 
                    scale_range=config.scale_range, 
                    height_shift_range=config.height_shift_range, 
                    width_shift_range=config.width_shift_range, 
                    erode_range=config.erode_range, 
                    dilate_range=config.dilate_range)
            x = pp.normalization(x)
            # y = [self.tokenizer.texts_to_sequences(word.decode())[0] for word in y]
            # y = np.array([np.pad(np.asarray(seq), (0, self.maxTextLength-len(seq)), constant_values=(-1, self.PAD)) for seq in y])
            y_ = []
            for line in y:
                seq = self.tokenizer.texts_to_sequences(line.decode())[0]
                padded_seq = np.pad(seq, (0, self.maxTextLength-len(seq)))
                y_.append(padded_seq)

            y = np.array(y_)

            return (x, y)

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        if self.partition in ['train'] and self.buf_size:
            self.img_buf = self.dataset['image'][0:self.buf_size]
            self.lab_buf = self.dataset['label'][0:self.buf_size]
