from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
# import lmdb
import io
import random
import json

import torch
from torch.utils.data import Dataset

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class FewShotDataset_test(Dataset):
    """Few shot epoish testing Dataset

    Returns a task (X_s, Y_s, Yt_s, X_q, Y_q, Yt_q) to classify'
        X_s: [n_way*n_shot, c, h, w].
        Y_s: [n_way*n_shot] from 0 to n_way-1
        Yt_s: [n_way*n_shot] from 0 to num_classes-1
        X_q:  [n_way*n_query, c, h, w].
        Y_q:  [n_way*n_query].
        Yt_q: [n_way*n_query].

    Difference between training dataset is that the episodics are sampled at the beginning so that in each iteration
    testing tasks remains the same.
    """

    def __init__(self,
                 data_path,
                 n_way=5, # number of categories.
                 n_shot=1, # number of training examples per category.
                 n_query=16, # number of testing examples per categories.
                 epoch_size=600, # number of tasks per epoch.
                 transform=None,
                 seed=0
                 ):
        self.data_path = data_path
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.transform = transform

        # self.nExemplars = nExemplars

        self.epoch_size = epoch_size
        self.seed = seed

        # load data file
        with open(data_path, 'r') as f:
            test_data_info = json.load(f)

        self.image_paths = test_data_info['image_names']
        self.image_labels = test_data_info['image_labels']

        self.labels2inds = {}
        for idx, label in enumerate(self.image_labels):
            if label not in self.labels2inds:
                self.labels2inds[label] = []
            self.labels2inds[label].append(idx)

        self.label_type = sorted(self.labels2inds.keys())

        # sample all testing tasks in the beginning
        random.seed(seed)
        np.random.seed(seed)

        self.Epoch_Support = []
        self.Epoch_Query = []
        for i in range(epoch_size):
            Query, Support = self._sample_episode()
            self.Epoch_Support.append(Support)
            self.Epoch_Query.append(Query)

    def __len__(self):
        return self.epoch_size

    def _sample_episode(self):
        """sampels a training epoish indexs.
        Returns:
            Query: a list of length 'n_way * n_query' with 3-element tuples. (sample_index, episodic_label, true_label)
            Support: a list of length 'n_way * n_shot' with 3-element tuples. (sample_index, episodic_label, true_label)
        """

        labels = random.sample(self.label_type, self.n_way)  # sample n_way labels
        n_way = len(labels)

        Query = []
        Support = []
        for label_idx in range(n_way):
            ids = (self.n_query + self.n_shot)  # for each class, we sample n_query+n_shot data
            img_ids = random.sample(self.labels2inds[labels[label_idx]], ids)

            imgs_query = img_ids[:self.n_query]
            imgs_support = img_ids[self.n_query:]

            Query.append([(img_id, label_idx, labels[label_idx]) for img_id in imgs_query])
            Support.append([(img_id, label_idx, labels[label_idx]) for img_id in imgs_support])

        assert(len(Query) == self.n_way)
        assert(len(Support) == self.n_way)
        assert(len(Query[0]) == self.n_query)
        assert(len(Support[0]) == self.n_shot)
        # random.shuffle(Query)
        # random.shuffle(Support)

        return Query, Support

    def _creatExamplesTensorData(self, examples):
        """
        Creats the examples image label tensor data.

        Args:
            examples: a list of list of 3-element tuples. (sample_index, label1, label2).

        Returns:
            images: a tensor [n_way, n_query, c, h, w]
            labels: a tensor [n_way, n_query]
            cls: a tensor [n_way, n_query]
        """

        images = []
        labels = []
        cls = []
        for class_info in examples: # all samples info from one class
            images_class = []
            labels_class = []
            cls_class = []
            for (img_idx, episodic_label, true_label) in class_info:
                img_pth = self.image_paths[img_idx]
                img = read_image(img_pth)
                if self.transform is not None:
                    img = self.transform(img)
                images_class.append(img)
                labels_class.append(episodic_label)
                cls_class.append(true_label)
            images_class = torch.stack(images_class, dim=0)
            labels_class = torch.LongTensor(labels_class)
            cls_class = torch.LongTensor(cls_class)
            images.append(images_class)
            labels.append(labels_class)
            cls.append(cls_class)
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        cls = torch.stack(cls, dim=0)
        return images, labels, cls

    def __getitem__(self, index):
        Query = self.Epoch_Query[index]
        Support = self.Epoch_Support[index]
        X_s, Y_s, Yt_s = self._creatExamplesTensorData(Support)
        X_q, Y_q, Yt_q = self._creatExamplesTensorData(Query)
        return X_s, Y_s, Yt_s, X_q, Y_q, Yt_q
