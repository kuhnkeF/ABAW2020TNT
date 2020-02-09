"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""

import cv2
import json
import os
from utils import get_filename

class Video():

    def __init__(self, path):
        self.path = path
        self.video = cv2.VideoCapture(path)
        self.frame_nr = 0
        self.json_file = path + 'meta.json'
        self.filename = get_filename(path)
        if os.path.isfile(self.json_file):
            with open(self.json_file, 'r') as config_file:
                self.meta = json.load(config_file)
        else:
            self.meta = {}
        if 'num_frames' in self.meta:
            self.num_frames = self.meta['num_frames']
        else:
            self.num_frames = self.count_frames()
            self.meta['num_frames'] = self.num_frames
            self.write_meta()

    def write_meta(self):
        with open(self.json_file, 'w') as outfile:
            json.dump(self.meta, outfile, indent=4)

    def is_ready(self):
        return self.video.isOpened()

    def rewind(self):
        self.video.release()
        self.frame_nr = 0
        self.video = cv2.VideoCapture(self.path)

    def read_BGR(self):
        is_good, image = self.video.read()
        if is_good:
            self.frame_nr += 1
        return is_good, image

    def read_RGB(self):
        is_good, image = self.read_BGR()
        return is_good, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def count_frames(self):
        if self.video.isOpened() == False:
            print("ERROR: Video File not open, can not count number of frames")
            return False
        total = 0
        print("Counting frames... takes some time")
        while True:
            # grab the current frame
            (grabbed, frame) = self.video.read()
            # check to see if we have reached the end of the
            # video
            if not grabbed:
                break
            # increment the total number of frames
            total += 1
            if total % 600 == 1:
                print(str(total))
        self.rewind()
        return total

    def release(self):
        self.video.release()

    def __len__(self):
        return self.num_frames

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame_nr >= self.num_frames:
            raise StopIteration
        good, frame = self.read_RGB()
        return frame
