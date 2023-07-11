import pickle
import torch

# with open('./encoded-data/bert-base-uncased/data_test/data_test/000.pkl', encoding='ascii') as f:
    # data = pickle.load(f)
    # print(data)
filepath = './encoded-data/bert-base-uncased/data_test/data_test/112.pkl'
filepath = './encoded-data/kobert/data_test/data_ko/000.pkl'

a = torch.load(filepath, map_location="cpu")
print(a)