import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]

class JapanItemsDataset(Dataset):

    def __init__(self, 
                imgs_path='data/downloaded_data/',
                anno_path='data/annotation/data.xlsx',
                is_train=True, 
                transform=None):

        self.imgs_path = imgs_path
        self.anno_path = anno_path
        self.transform = transform
        self.is_train = is_train

        # Loading pandas dataframe annotation
        self.anno, self.classes = self._load_anno(self.anno_path)
        self.data = self._train_test_split(self.anno)

    def _read_image(self, img_path):
        return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    def _load_anno(self, anno_path):
        anno = pd.read_excel(anno_path)

        # Transform relative image path to absolute
        anno.iloc[:, 0] = anno.iloc[:, 0].apply(lambda img: self.imgs_path + img)

        # Label encodings
        anno.iloc[:, 1] = anno.iloc[:, 1].astype('category')
        anno['labels'] = anno.iloc[:, 1].cat.codes

        # Dict of classes
        classes = dict(enumerate(anno.iloc[:, 1].cat.categories))

        return anno, classes
    
    def _train_test_split(self, anno, test_size=0.1):
        X, y = anno.iloc[:, 0].to_numpy(), anno.iloc[:, 2].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=test_size, 
                                                            shuffle=True, 
                                                            stratify=y, 
                                                            random_state=42)

        return (X_train, y_train) if self.is_train else (X_test, y_test)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        
        image = self._read_image(self.data[0][idx])
        target = torch.tensor(self.data[1][idx], dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, target


if __name__ == '__main__':

    from transforms import *

    dataset = JapanItemsDataset(imgs_path='data/downloaded_data/',
                                anno_path='data/annotation/data.xlsx',
                                is_train=True, 
                                transform=train_transforms())

    # Transform from tensor to numpy array
    def transform_img(img):
        img = img.numpy()
        img = img.transpose((1, 2, 0))
        img = STD * img + MEAN
        img = np.clip(img, 0, 1)
        return img

    out_dir = str(ROOT / 'data/dataset_data')
    os.makedirs(out_dir, exist_ok=True)

    for idx, (image, target) in tqdm(enumerate(dataset)):
        image = transform_img(image)
        target = int(target.numpy())
        out_fpath = out_dir + '/' + str(idx) + '.jpg'

        plt.imshow(image)
        plt.title(dataset.classes[target])
        plt.savefig(out_fpath)

        if idx >= 20:
            break

    print('Done!')
