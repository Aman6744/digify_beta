import numpy as np
import os
import multiprocessing
import h5py

from tqdm import tqdm
from glob import glob
from functools import partial
from data import data_preprocessor as pp
from config import config

class Sample():
    def __init__(self, file_path, label):
        self.file_path = file_path
        self.label = label

class Dataset():
    def __init__(self):

        assert os.path.exists(config.raw_path)

        self.dataset = {
                'train': {
                    'image':[],
                    'label':[], 
                    'augmentation':True
                    }, 
                'test': {
                    'image':[], 
                    'label':[], 
                    'augmentation':False
                    },
                'valid': {
                    'image':[], 
                    'label':[], 
                    'augmentation':False
                    }
                }
        self.samples = []
        self.raw_path = config.raw_path
        self.imgdirs = [
        	os.path.join(self.raw_path, "IAM", "words"), 
        	os.path.join(self.raw_path, "IAM", "lines"), 
        	os.path.join(self.raw_path, "BENTHAM", "lines"), 
        	]
        self.label_paths = [
        	os.path.join(self.raw_path, "IAM", "words.txt"), 
        	os.path.join(self.raw_path, "IAM", "lines.txt"), 
        	os.path.join(self.raw_path, "BENTHAM", "transcriptions")]
        self.partitions = ['train', 'test', 'valid']
        self.maxTextLength = config.maxTextLength
        self.target = config.source_path
        self.target_image_size = config.target_image_size


    def make_partitions(self):
        for idx in [1]:
            self.make_partitions_helper_IAM(idx)
    	
        # self.make_partitions_helper_BENTHAM()

        np.random.shuffle(self.samples)

        print(f"Data ready for making partitions!!\nTotal_items: {len(self.samples)}")

        splitIdx1 = {'train':0, 'test':0, 'valid':0}
        splitIdx2 = {'train':0, 'test':0, 'valid':0}
        splitIdx1['test'] = splitIdx2['train'] = int(0.8 * len(self.samples))
        splitIdx2['test'] = splitIdx1['valid'] = int(0.9 * len(self.samples))
        splitIdx2['valid'] = len(self.samples)

        for p in self.partitions:
            self.dataset[p]['image'] += [sample.file_path for sample in self.samples[splitIdx1[p]:splitIdx2[p]]]
            self.dataset[p]['label'] += [sample.label for sample in self.samples[splitIdx1[p]:splitIdx2[p]]]
        print(f"\nTrain, test, valid partitions made!!")
        for p in self.partitions:
            print(f"{p} size: {len(self.dataset[p]['image'])}")

    def make_partitions_helper_IAM(self, index):
        variable = "lines" if index==1 else "words"
        print(f"Transforming the IAM {variable} Dataset!!")
        label = open(self.label_paths[index]).read().splitlines()
        for line in label:
            if line[0] == '#' or not line:
                continue
            lineSplit = line.strip().split()
            if len(lineSplit) < 9 or lineSplit[1] == "err":
                continue
            transcription = pp.preprocess_label("|".join(lineSplit[8:]).replace("|", " "), self.maxTextLength)
            if transcription[0]:
                transcription = transcription[1]
            else:
                continue
            fileNameSplit = lineSplit[0].split('-')
            fileName = os.path.join(self.imgdirs[index], fileNameSplit[0], "-".join(fileNameSplit[0:2]), lineSplit[0]) + ".png"
            sample = Sample(fileName, transcription)
            self.samples.append(sample)

        if index == 0:
            np.random.shuffle(self.samples)
            self.samples = self.samples[:7000]

    def make_partitions_helper_BENTHAM(self):
        print(f"Transforming the BENTHAM lines Dataset!!")
        imgdir = self.imgdirs[2]
        label_path = self.label_paths[2]

        labels = sorted(glob(os.path.join(label_path, "*.txt")))
        for label in labels:
            fileName = os.path.join(imgdir, label.split("/")[-1].split(".")[0]+".png")
            with open(label, "r") as f:
                transcription = f.read().strip().split()
            
            transcription = pp.preprocess_label(" ".join(transcription), self.maxTextLength)
            if transcription[0]:
                transcription = transcription[1]
            else:
                continue

            sample = Sample(fileName, transcription)
            self.samples.append(sample)

    def save_partitions(self):
        self.make_partitions()

        os.makedirs(os.path.dirname(self.target), exist_ok=True)
        total = 0

        with h5py.File(self.target, 'w') as hf:
            for p in self.partitions:
                size = (len(self.dataset[p]['image']), ) + self.target_image_size[:2]
                total += size[0]

                hf.create_dataset(f"{p}/image", size, dtype=np.uint8, compression='gzip', compression_opts=9)
                hf.create_dataset(f"{p}/label", (size[0],), dtype=f"S{self.maxTextLength}", compression='gzip', compression_opts=9)

        pbar = tqdm(total=total)
        batch_size = 1024

        for p in self.partitions:
            for batch in range(0, len(self.dataset[p]['image']), batch_size):
                images = []

                with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                    r = pool.map(partial(pp.preprocess_image, target_size=self.target_image_size, augmentation=self.dataset[p]['augmentation']), self.dataset[p]['image'][batch:batch+batch_size])
                    images.append(r)
                    pool.close()
                    pool.join()

                with h5py.File(self.target, "a") as hf:
                    hf[f"{p}/image"][batch:batch+batch_size] = images
                    hf[f"{p}/label"][batch:batch+batch_size] = [s.encode() for s in self.dataset[p]['label'][batch:batch+batch_size]]
                    pbar.update(batch_size)

