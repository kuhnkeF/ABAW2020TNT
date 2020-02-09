"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tsav import TwoStreamAuralVisualModel
from aff2compdataset import Aff2CompDataset
from write_labelfile import write_labelfile
from utils import ex_from_one_hot, split_EX_VA_AU
from tqdm import tqdm
import os

model_path = 'TSAV416k.pth.tar' # path to the model
result_path = 'results'# path where the result .txt files should be stored
database_path = 'aff2_processed/'  # path where the database was created (images, audio...) see create_database.py
# should be the same path


class SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)


    def __len__(self):
        return len(self.indices)

if __name__ == '__main__':

    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        cudnn.enabled = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print('cpu selected!')
    # model
    model = TwoStreamAuralVisualModel(num_channels=4)
    modes = model.modes
    # load the model
    saved_model = torch.load(model_path, map_location=device)
    model.load_state_dict(saved_model['state_dict'])
    model = model.to(device)
    # disable grad, set to eval
    for p in model.parameters():
        p.requires_grad = False
    for p in model.children():
        p.train(False)

    # load dataset (first time this takes longer)
    dataset = Aff2CompDataset(database_path)
    dataset.set_modes(modes)

    # select the frames we want to process (we choose VAL and TEST)
    testvalids = np.logical_or(dataset.test_ids, dataset.val_ids)
    print('Validation set length: ' + str(sum(dataset.val_ids)))
    print('Test set length: ' + str(sum(dataset.test_ids)))
    sampler = SubsetSequentialSampler(np.nonzero(testvalids)[0])
    loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=8, pin_memory=False, drop_last=False)

    output = torch.zeros((len(dataset), 17), dtype=torch.float32)
    #labels = torch.zeros((len(dataset), 17), dtype=torch.float32)

    # run inference
    # takes 5+ hours for test and val on 2080 Ti, with data on ssd
    for data in tqdm(loader):
        # ex_label = data['EX'].float()
        # va_label = data['VA'].float()
        # au_label = data['AU'].float()
        ids = data['Index'].long()

        x = {}
        for mode in modes:
            x[mode] = data[mode].to(device)

        result = model(x)
        output[ids, :] = result.detach().cpu()  # output is EX VA AU
        #labels[ids, :] = torch.cat([ex_label, va_label, au_label], dim=1)

    # store the predictions so we can skip inference later
    torch.save({'predictions': output}, os.path.join(result_path, 'inference.pkl'))

    # load the predictions
    #output = torch.load(os.path.join(result_path, 'inference.pkl')['predictions']

    # VALIDATION RESULTS
    # export the results as txt files
    print('writing text files validation')
    o_p_v = os.path.join(result_path, 'val')
    for i in range(len(dataset.val_video_indices)):
        # should produce 145 files
        print(i)
        indices = dataset.val_video_indices[i]
        name_pos = dataset.val_video_real_names[i]
        types = dataset.val_video_types[i]
        EX, VA, AU = split_EX_VA_AU(output[indices])
        print(name_pos)
        print('With ' + str(EX.shape[0]) + " entries")
        if 'AU' in types:
            write_labelfile(AU.numpy(), 'AU', name_pos, position_str=None, result_dir=o_p_v)
        if 'VA' in types:
            write_labelfile(VA.numpy(), 'VA', name_pos, position_str=None, result_dir=o_p_v)
        if 'EX' in types:
            write_labelfile(ex_from_one_hot(EX.numpy()), 'EX', name_pos, position_str=None, result_dir=o_p_v)
    print('done val')

    # TEST RESULTS
    o_p_t = os.path.join(result_path, 'test')
    print('writing text files test')
    for i in range(len(dataset.test_video_indices)):
        # 14 AU
        # 223 EX
        # 139 VA
        # = 376
        print(i)
        indices = dataset.test_video_indices[i]
        name_pos = dataset.test_video_real_names[i]
        types = dataset.test_video_types[i]
        EX, VA, AU = split_EX_VA_AU(output[indices])
        print(name_pos)
        print('With ' + str(EX.shape[0]) + " entries")
        if 'AU' in types:
            write_labelfile(AU.numpy(), 'AU', name_pos, position_str=None, result_dir=o_p_t)
        if 'VA' in types:
            write_labelfile(VA.numpy(), 'VA', name_pos, position_str=None, result_dir=o_p_t)
        if 'EX' in types:
            write_labelfile(ex_from_one_hot(EX.numpy()), 'EX', name_pos, position_str=None, result_dir=o_p_t)
    print('done test')