import torch
import numpy
from torch.utils.data import TensorDataset, DataLoader


def return_loader(dataset, batch_size=8):

    sequences, labels = dataset

    x, y, z = [], [], []

    for sequence, label in zip(sequences, labels):
        x.append(sequence)
        y.append(label)
        z.append(np.count_nonzero(sequence))

    tensor_dataset = TensorDataset(torch.tensor(x).long(),
                                   torch.tensor(z).long(),
                                   torch.tensor(y).long())

    return DataLoader(tensor_dataset, batch_size=batch_size, drop_last=True)