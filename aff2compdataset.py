"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import pickle
import torchaudio
import math
import subprocess
from utils import *
from clip_transforms import *
from video import Video
from torch.utils.data import Dataset

class Aff2CompDataset(Dataset):
    def __init__(self, root_dir=''):
        super(Aff2CompDataset, self).__init__()
        self.video_dir = root_dir
        self.extracted_dir = os.path.join(self.video_dir, 'extracted')

        self.clip_len = 8
        self.input_size = (112, 112)
        self.dilation = 6
        self.label_frame = self.clip_len * self.dilation

        # audio params
        self.window_size = 20e-3
        self.window_stride = 10e-3
        self.sample_rate = 44100
        num_fft = 2 ** math.ceil(math.log2(self.window_size * self.sample_rate))
        window_fn = torch.hann_window

        self.sample_len_secs = 10
        self.sample_len_frames = self.sample_len_secs * self.sample_rate
        self.audio_shift_sec = 5
        self.audio_shift_samples = self.audio_shift_sec * self.sample_rate
        # transforms

        self.audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_mels=64,
                                                                    n_fft=num_fft,
                                                                    win_length=int(self.window_size * self.sample_rate),
                                                                    hop_length=int(self.window_stride
                                                                                   * self.sample_rate),
                                                                    window_fn=window_fn)

        self.audio_spec_transform = ComposeWithInvert([AmpToDB(), Normalize(mean=[-14.8], std=[19.895])])
        self.clip_transform = ComposeWithInvert([NumpyToTensor(), Normalize(mean=[0.43216, 0.394666, 0.37645, 0.5],
                                                                            std=[0.22803, 0.22145, 0.216989, 0.225])])


        all_videos = find_all_video_files(self.video_dir)
        self.cached_metadata_path = os.path.join(self.video_dir, 'dataset.pkl')

        if not os.path.isfile(self.cached_metadata_path):
            self.image_path = []  # paths relative to self.extracted_dir
            self.video_id = []
            self.frame_id = []
            self.label_au = []
            self.label_ex = []
            self.label_va = []
            self.train_ids = []
            self.val_ids = []
            self.test_ids = []
            self.features = []
            self.feature_names = []
            self.time_stamps = []
            self.mask_available = False
            for video in tqdm(all_videos):
                meta = Video(video).meta
                meta['filename'] = get_filename(video)
                meta['path'] = get_path(video)
                meta['extension'] = get_extension(video)
                num_frames_video = meta['num_frames']
                audio_file = os.path.splitext(video)[0] + '.wav'
                si, ei = torchaudio.info(audio_file)
                assert si.rate == 44100
                video_ts_file = os.path.join(meta['path'], meta['filename'] + '_video_ts.txt')
                if os.path.isfile(video_ts_file):
                    pass
                else:
                    mkvfile = os.path.join(meta['path'], 'temp.mkv')
                    videofile = os.path.join(meta['path'], meta['filename'] + meta['extension'])
                    command = 'mkvmerge -o ' + mkvfile + ' ' + videofile
                    subprocess.call(command, shell=True)
                    command = 'mkvextract ' + mkvfile + ' timestamps_v2 0:' + video_ts_file
                    subprocess.call(command, shell=True)
                    os.remove(mkvfile)
                with open(video_ts_file, 'r') as f:
                    time_stamps = np.genfromtxt(f)[:num_frames_video]
                #os.remove(video_ts_file)
                self.mask_available = True

                extracted_dir = os.path.join(self.extracted_dir, meta['filename'])


                splits = []
                if 'AU' in meta:
                    au_split = meta['AU']
                    splits.append(au_split)
                if 'EX' in meta:
                    ex_split = meta['EX']
                    splits.append(ex_split)
                if 'VA' in meta:
                    va_split = meta['VA']
                    splits.append(va_split)

                for split in splits:
                    self.time_stamps.append(time_stamps)
                    for image_filename in sorted(os.listdir(extracted_dir)):
                        if os.path.isdir(os.path.join(extracted_dir, image_filename)):
                            continue
                        # path relative to self.extracted_dir
                        self.image_path.append(os.path.relpath(os.path.join(extracted_dir, image_filename), self.extracted_dir))
                        self.video_id.append(meta['filename'])
                        frame_id = int(os.path.splitext(image_filename)[0])
                        self.frame_id.append(frame_id)
                        # add your own label loading here if you want to use this for training
                        self.label_au.append(None)
                        self.label_ex.append(None)
                        self.label_va.append(None)
                        self.train_ids.append(1 if split == 'train' else 0)
                        self.val_ids.append(1 if split == 'val' else 0)
                        self.test_ids.append(1 if split == 'test' else 0)


            self.frame_id = np.stack(self.frame_id)
            self.label_au = np.stack(self.label_au)
            self.label_ex = np.stack(self.label_ex)
            self.label_va = np.stack(self.label_va)
            self.train_ids = np.stack(self.train_ids)
            self.val_ids = np.stack(self.val_ids)
            self.test_ids = np.stack(self.test_ids)
            self.time_stamps = np.hstack(self.time_stamps)

            with open(self.cached_metadata_path, 'wb') as f:
                pickle.dump({'frame_id': self.frame_id,
                             'label_au': self.label_au,
                             'label_ex': self.label_ex,
                             'label_va': self.label_va,
                             'video_id': self.video_id,
                             'image_path': self.image_path,
                             'train_ids': self.train_ids,
                             'val_ids': self.val_ids,
                             'test_ids': self.test_ids,
                             'time_stamps': self.time_stamps,
                             'mask_available': self.mask_available}, f)
        else:
            with open(self.cached_metadata_path, 'rb') as f:
                meta = pickle.load(f)
                self.frame_id = meta['frame_id']
                self.label_au = meta['label_au']
                self.label_ex = meta['label_ex']
                self.label_va = meta['label_va']
                self.video_id = meta['video_id']
                self.image_path = meta['image_path']
                self.train_ids = meta['train_ids']
                self.val_ids = meta['val_ids']
                self.time_stamps = meta['time_stamps']
                self.mask_available = meta['mask_available']
                self.test_ids = meta['test_ids']

        self.validation_video_ids()
        self.test_video_ids()
        self.use_mask = self.mask_available

    def validation_video_ids(self):
        # return ids that belong to validation videos, regardless if the labels are valid or not
        print('val ids...')
        val_indices = np.nonzero(self.val_ids == 1)[0]
        val_video_id = []
        for i in val_indices:
            val_video_id.append(self.video_id[i])
        self.unique_val_videos = sorted(list(set(val_video_id))) # val should be 145 videos
        assert len(self.unique_val_videos) == 145
        self.val_video_indices = []
        self.val_video_real_names = []
        self.val_video_types = []
        for vid in self.unique_val_videos:
            tmp = [i for i in range(len(self.video_id)) if self.video_id[i] == vid and self.val_ids[i] == 1]
            self.val_video_indices.append(tmp)
            # append the "real name
            position = get_position(vid)
            vid_path = os.path.join(self.video_dir, vid + '.mp4')
            vid_path2 = os.path.join(self.video_dir, vid + '.avi')
            if os.path.isfile(vid_path):
                real_name = Video(vid_path).meta['original_video']
            elif os.path.isfile(vid_path2):
                real_name = Video(vid_path2).meta['original_video']
            else:
                print(vid_path)
                print(vid_path2)
                raise NameError('video not found')
            self.val_video_real_names.append(real_name + position)
            # check what types should be processed for this video
            self.val_video_types.append(self.get_types_from_name_split(vid, 'val'))

    def test_video_ids(self):
        # return ids that belong to validation videos, regardless if the labels are valid or not
        print('test ids...')
        test_indices = np.nonzero(self.test_ids == 1)[0]
        test_video_id = []
        for i in test_indices:
            test_video_id.append(self.video_id[i])

        self.unique_test_videos = sorted(list(set(test_video_id)))
        self.test_video_indices = []
        self.test_video_real_names = []
        self.test_video_types = []
        all = 0
        for vid in self.unique_test_videos:
            tmp = [i for i in range(len(self.video_id)) if self.video_id[i] == vid and self.test_ids[i] == 1]
            self.test_video_indices.append(tmp)
            # append the "real name
            all += len(tmp)
            position = get_position(vid)
            vid_path = os.path.join(self.video_dir, vid + '.mp4')
            vid_path2 = os.path.join(self.video_dir, vid + '.avi')
            if os.path.isfile(vid_path):
                real_name = get_filename(os.path.realpath(vid_path))
            elif os.path.isfile(vid_path2):
                real_name = get_filename(os.path.realpath(vid_path2))
            else:
                print(vid_path)
                print(vid_path2)
                raise NameError('video not found')
            self.test_video_real_names.append(real_name + position)
            # check what types should be processed for this video
            self.test_video_types.append(self.get_types_from_name_split(vid, 'test'))

    def get_types_from_name_split(self, name, split):
        au = name[6:8]
        va = name[16:18]
        ex = name[11:13]
        test = []
        train = []
        val = []
        if au[0] == '1':
            if au[1] == '_':
                train.append('AU')
            if au[1] == 'v':
                val.append('AU')
            if au[1] == 't':
                test.append('AU')
        if va[0] == '1':
            if va[1] == '_':
                train.append('VA')
            if va[1] == 'v':
                val.append('VA')
            if va[1] == 't':
                test.append('VA')
        if ex[0] == '1':
            if ex[1] == '_':
                train.append('EX')
            if ex[1] == 'v':
                val.append('EX')
            if ex[1] == 't':
                test.append('EX')
        if split == 'train':
            return train
        if split == 'val':
            return val
        if split == 'test':
            return test

    def set_clip_len(self, clip_len):
        assert(np.mod(clip_len, 2) == 0)  # clip length should be even at this point
        self.clip_len = clip_len

    def set_modes(self, modes):
        self.modes = modes

    def __getitem__(self, index):

        # get labels, convert it to one-hot encoding etc.
        # ...
        # compute pseudo-labels for ex and va using the distribution of ex- and va-labels
        # ...
        # original code
        # data = {'AU': None,
        #         'EX': None,
        #         'VA': None,
        #         'Index': index}
        # why is this not working?
        data = {'Index': index}

        video_id = self.video_id[index]

        if self.use_mask:
            clip = np.zeros((self.clip_len, self.input_size[0], self.input_size[1], 4), dtype=np.uint8)
            # init all frames black
        else:
            clip = np.zeros((self.clip_len, self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        _range = range(index - self.label_frame + self.dilation,
                       index - self.label_frame + self.dilation * (self.clip_len + 1), self.dilation)
        for clip_i, all_i in enumerate(_range):
            if all_i < 0 or all_i >= len(self) or self.video_id[all_i] != video_id:
                # leave frame black
                continue
            else:
                img = Image.open(os.path.join(self.extracted_dir, self.image_path[all_i]))
                if self.use_mask:
                    mask_img = Image.open(os.path.join(self.extracted_dir, os.path.dirname(self.image_path[all_i]),
                                                       'mask', os.path.basename(self.image_path[all_i])))
                try:
                    if self.use_mask:
                        clip[clip_i, :, :, 0:3] = np.array(img)
                        clip[clip_i, :, :, 3] = np.array(mask_img)
                    else:
                        clip[clip_i] = np.array(img)
                except:
                    # loading an image fails leave that frame black
                    print(os.path.join(self.extracted_dir, self.image_path[all_i]))
        data['clip'] = self.clip_transform(clip)

        # get audio
        audio_file = os.path.join(self.video_dir, self.video_id[index] + '.wav')

        audio, sample_rate = torchaudio.load(audio_file,
                                             num_frames=min(self.sample_len_frames,
                                                            max(int((self.time_stamps[index]/1000) * self.sample_rate),
                                                                int(self.window_size * self.sample_rate))),
                                             offset=max(int((self.time_stamps[index]/1000) * self.sample_rate
                                                            - self.sample_len_frames + self.audio_shift_samples), 0))


        audio_features = self.audio_transform(audio).detach()
        if audio.shape[1] < self.sample_len_frames:
            _audio_features = torch.zeros((audio_features.shape[0], audio_features.shape[1],
                                           int((self.sample_len_secs / self.window_stride) + 1)))
            _audio_features[:, :, -audio_features.shape[2]:] = audio_features
            audio_features = _audio_features

        if self.audio_spec_transform is not None:
            audio_features = self.audio_spec_transform(audio_features)

        data['audio_features'] = audio_features

        if audio.shape[1] < self.sample_len_frames:
            _audio = torch.zeros((1, self.sample_len_frames))
            _audio[:, -audio.shape[1]:] = audio
            audio = _audio
        data['audio'] = audio

        return data

    def __len__(self):
        return len(self.video_id)

    def __add__(self, other):
        raise NotImplementedError
