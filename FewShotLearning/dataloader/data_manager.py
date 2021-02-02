from __future__ import absolute_import
from __future__ import print_function

import torchvision.transforms as T
from .data_loader import FewShotDataset_test
from torch.utils.data import DataLoader

class FewShotDataManager(object):
    def __init__(self, data_path, image_size, n_way, n_shot, n_query, epoch_size, batch_size, test_seed):

        data_transform = T.Compose([
            T.Resize(int(image_size*1.1)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([x / 255.0 for x in [125.3, 123.0, 113.9]], [x / 255.0 for x in [63.0, 62.1, 66.7]])])

        self.dataset = FewShotDataset_test(data_path, n_way, n_shot, n_query, epoch_size, data_transform, test_seed)
        print('{} cats, {} images.'.format(len(self.dataset.label_type), len(self.dataset.image_paths)))
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

