# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import glob
import torch
import random
import json
import csv
import numpy as np
import numpy.random
import logging
from collections import defaultdict
import torch.distributed as dist

from src import dist_utils


data_path = './encoded-data/bert-base-uncased/data_test/data_test'
files = glob.glob(os.path.join(data_path, "*.p*"))
print(files)
files.sort()
tensors = []
files_split = list(np.array_split(files, dist_utils.get_world_size()))[dist_utils.get_rank()]

print(files_split)