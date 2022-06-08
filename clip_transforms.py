"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import torch
from torchvision.transforms.functional import normalize
import numpy as np
import cv2
import random
import importlib
from torchaudio.transforms import AmplitudeToDB


class ComposeWithInvert(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, invert=False):
        if invert:
            for t in reversed(self.transforms):
                img = t(img, invert)
        else:
            for t in self.transforms:
                img = t(img, invert)
        return img


class NumpyToTensor:
    #convert numpy to tensor, or tensor to tensor
    def __init__(self):
        pass

    def __call__(self, clip, invert):

        if invert:
            #convert to TODO
            clip = clip.permute(1, 2, 3, 0)
            clip = clip.mul(255).to(torch.uint8)
        else:
            #convert from img int8 T, W, H, C to float 0-1 image C T W H
            clip = clip.astype(np.float32) / 255
            clip = torch.from_numpy(clip).permute(3, 0, 1, 2)

        return clip


class Normalize:
    """Normalize an tensor image or video clip with mean and standard deviation.
       Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.

        # forward is an in place operation!
        # invert is not
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.mean_t = None
        self.std_t = None

    def __call__(self, clip, invert):
        if self.mean_t is None:
            dtype = clip.dtype
            if len(clip.shape) == 4:
                self.mean_t = torch.as_tensor(self.mean, dtype=dtype, device=clip.device)[:, None, None, None]
                self.std_t = torch.as_tensor(self.std, dtype=dtype, device=clip.device)[:, None, None, None]
            else:
                self.mean_t = torch.as_tensor(self.mean, dtype=dtype, device=clip.device)[:, None, None]
                self.std_t = torch.as_tensor(self.std, dtype=dtype, device=clip.device)[:, None, None]

        if invert:
            clip = clip.clone()
            clip.mul_(self.std_t).add_(self.mean_t)
        else:
            # image of size (C, H, W) to be normalized.
            #clip = normalize(clip, self.mean, self.std)
            clip.sub_(self.mean_t).div_(self.std_t)

        return clip


class AmpToDB:

    def __init__(self):
        self.amplitude_to_DB = AmplitudeToDB('power', 80)

    def __call__(self, features, invert):

        if invert:
            pass # do nothing
        else:
            features = self.amplitude_to_DB(features)

        return features


class RandomClipFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip, invert):

        if invert:
            pass # do nothing
        else:
            if random.random() < self.p:
                # T W H C
                #assert clip.shape[3] == 3 # last channel is RGB
                # for every image apply cv2 flip
                for i in range(clip.shape[0]):
                    clip[i] = cv2.flip(clip[i], 1)

        return clip

class HSVJitter():

    def __init__(self, h_range=10, s_range=25, v_range=20):
        self.h_s_v_range = (h_range, s_range, v_range)

    @staticmethod
    def shift_hsv_uint8(img, hsv_shift):
        # adaptated and modified from albumination (check albumentations.ai)
        dtype = img.dtype
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue, sat, val = cv2.split(img)

        lut_hue = np.arange(0, 256, dtype=np.int16)
        lut_hue = np.mod(lut_hue + hsv_shift[0], 180).astype(dtype)

        lut_sat = np.arange(0, 256, dtype=np.int16)
        lut_sat = np.clip(lut_sat + hsv_shift[1], 0, 255).astype(dtype)

        lut_val = np.arange(0, 256, dtype=np.int16)
        lut_val = np.clip(lut_val + hsv_shift[2], 0, 255).astype(dtype)

        hue = cv2.LUT(hue, lut_hue)
        sat = cv2.LUT(sat, lut_sat)
        val = cv2.LUT(val, lut_val)

        img = cv2.merge((hue, sat, val)).astype(dtype)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    def __call__(self, clip):
           # T W H C
           # only works with uint8 images
        hsv_shift = np.zeros((3,1), dtype=np.int16)
        for i in range(3):
            hsv_shift[i] = random.uniform(-self.h_s_v_range[i], self.h_s_v_range[i])

        for i in range(clip.shape[0]):
            clip[i, :, :, 0:3] = self.shift_hsv_uint8(np.squeeze(clip[i, :, :, 0:3]), hsv_shift)

        return clip
