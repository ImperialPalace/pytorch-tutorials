#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/8/9
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-

# In[0]
from torch.utils.data import Dataset, DataLoader


class MapDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
map_dataset = MapDataset(data)
loader = DataLoader(map_dataset, batch_size=4)
for batch in loader:
    print(batch)

# In[1]
from torch.utils.data import IterableDataset, DataLoader


class CustomIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)


data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
map_dataset = CustomIterableDataset(data)
loader = DataLoader(map_dataset, batch_size=4)
for batch in loader:
    print(batch)

# In[2]
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle


class MyIterableDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def parse_file(self, file_path):
        with open(file_path, 'r') as file_obj:
            for line in file_obj:
                tokens = line.strip('\n').split(' ')
                yield from tokens

    def get_stream(self, file_path):
        return cycle(self.parse_file(file_path))

    def __iter__(self):
        return self.get_stream(self.file_path)


iterable_dataset = MyIterableDataset("/work/pytorch-tutorials/example-08/input_data/file.txt")
loader = DataLoader(iterable_dataset, batch_size=5)

for batch in loader:
    print(batch)

# In[3]
from torch.utils.data import IterableDataset, DataLoader
from itertools import islice


class MyIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def process_data(self, data):
        for x in data:
            yield x

    def get_stream(self, data):
        return cycle(self.process_data(data))

    def __iter__(self):
        return self.get_stream(self.data)


data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
iterable_dataset = MyIterableDataset(data)
loader = DataLoader(iterable_dataset, batch_size=4)

for batch in islice(loader, 10):
    print(batch)

# In[4]
from torch.utils.data import IterableDataset, DataLoader
from itertools import islice, chain


class MyIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def process_data(self, data):
        for x in data:
            yield x

    def get_stream(self, data):
        return chain.from_iterable(map(self.process_data, cycle(data)))

    def __iter__(self):
        return self.get_stream(self.data)

data = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [10, 11, 12, 13, 14, 15, 16, 17, 18], [70, 71, 72, 73]]
iterable_dataset = MyIterableDataset(data)
loader = DataLoader(iterable_dataset, batch_size=4)

for batch in islice(loader, 10):
    print(batch)
