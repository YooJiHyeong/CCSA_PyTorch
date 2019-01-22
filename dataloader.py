import random
import numpy as np

import torch
from torch.utils.data import Dataset


# Initialization.Create_Pairs
class TrainSet(Dataset):
    def __init__(self, domain_adaptation_task, repetition, sample_per_class):
        x_source_path = './row_data/' + domain_adaptation_task + '_X_train_source_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'
        y_source_path = './row_data/' + domain_adaptation_task + '_y_train_source_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'
        x_target_path = './row_data/' + domain_adaptation_task + '_X_train_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'
        y_target_path = './row_data/' + domain_adaptation_task + '_y_train_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class) + '.npy'

        self.x_source=np.load(x_source_path)
        self.y_source=np.load(y_source_path)
        self.x_target=np.load(x_target_path)
        self.y_target=np.load(y_target_path)

        print("Source X : ", len(self.x_source), " Y : ", len(self.y_source))
        print("Target X : ", len(self.x_target), " Y : ", len(self.y_target))
                
        Training_P=[]
        Training_N=[]
        for trs in range(len(self.y_source)):
            for trt in range(len(self.y_target)):
                if self.y_source[trs] == self.y_target[trt]:
                    Training_P.append([trs,trt, 1])
                else:
                    Training_N.append([trs,trt, 0])
        print("Class P : ", len(Training_P), " N : ", len(Training_N))
        
        random.shuffle(Training_N)
        self.imgs = Training_P+Training_N[:3*len(Training_P)]
        random.shuffle(self.imgs)

    def __getitem__(self, idx):
        src_idx, tgt_idx, domain = self.imgs[idx]

        x_src, y_src = self.x_source[src_idx], self.y_source[src_idx]
        x_tgt, y_tgt = self.x_target[tgt_idx], self.y_target[tgt_idx]

        x_src = torch.from_numpy(x_src).unsqueeze(0)
        x_tgt = torch.from_numpy(x_tgt).unsqueeze(0)

        return x_src, y_src, x_tgt, y_tgt

    def __len__(self):
        return len(self.imgs)


class TestSet(Dataset):
    def __init__(self, domain_adaptation_task, repetition, sample_per_class):
        self.x_test = np.load('./row_data/' + domain_adaptation_task + '_X_test_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class)+'.npy')
        self.y_test = np.load('./row_data/' + domain_adaptation_task + '_y_test_target_repetition_' + str(repetition) + '_sample_per_class_' + str(sample_per_class)+'.npy')

    def __getitem__(self, idx):
        x, y = self.x_test[idx], self.y_test[idx]
        x = torch.from_numpy(x).unsqueeze(0)
        return x, y

    def __len__(self):
        return len(self.x_test)
