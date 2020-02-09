"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import os, glob
import numpy as np

def split_EX_VA_AU(inp):
    EX = inp[:, 0:7]
    VA = inp[:, 7:9]
    AU = inp[:, 9:]
    return EX, VA, AU

def ex_from_one_hot(ex_arr):
    assert isinstance(ex_arr, np.ndarray)
    assert ex_arr.shape[1] == 7
    len_ex_arr = ex_arr.shape[0]

    if len_ex_arr > 1:
        ex_label = np.zeros(len_ex_arr, dtype=np.int)
        for i in range(len_ex_arr):
            ex_label[i] = np.argmax(ex_arr[i])
    else:
        ex_label = np.zeros(len_ex_arr, dtype=np.int)
        ex_label[0] = np.argmax(ex_arr)

    return ex_label

def get_filename(n):
    filename, ext = os.path.splitext(os.path.basename(n))
    return filename

def get_extension(n):
    filename, ext = os.path.splitext(os.path.basename(n))
    return ext

def get_path(n):
    head, tail = os.path.split(n)
    return head

def convert_to_filenames(path_list, sort_list=True):
    str_list = []
    for path in path_list:
        str_list.append(get_filename(path))
    if sort_list:
        str_list.sort()
    return str_list

def solve_symlinks(path_list):
    str_list = []
    for path in path_list:
        str_list.append(os.path.realpath(path))
    return str_list

def get_position(name):
    if name[-5:] == "_main":
        position = "_main"
    elif name[-5:] == "_left":
        position = "_left"
    elif name[-6:] == "_right":
        position = "_right"
    else:
        position = ""
    return position

def find_all_files_with_ext_in(folder, ext):
    if ext[0] == ".":
        ext = "*"+ext
    else:
        ext = "*."+ext
    str_list = glob.glob(os.path.join(folder, ext))
    str_list.sort() # other way would be to use sorted but this should be faster
    return str_list

def find_all_video_files(folder):
    ext = ["avi", "AVI", "MP4", "mp4", "mkv", "MKV", "MOV", "mov", "WMV", "wmv", "webm", "WEBM", "mpg", "mpeg", "MPG", "MPEG"]
    str_list = []
    for t in ext:
        str_list += glob.glob(os.path.join(folder, "*."+t))
    str_list.sort()
    return str_list

def find_all_image_files(folder):
    ext = ["bmp", "jpg", "png", "PNG", "JPEG", "JPG", "jpeg", "tif", "tiff", "tga"]
    str_list = []
    for t in ext:
        str_list += glob.glob(os.path.join(folder, "*."+t))
    str_list.sort()
    return str_list

def get_label_str2(data):
    labels = {'AU': '0_',
              'EX': '0_',
              'VA': '0_'}
    for type in data:
        split = data[type]['original_split']
        if split == 'train':
            labels[type] = '1_'
        elif split == 'val':
            labels[type] = '1v'
        elif split == 'test':
            labels[type] = '1t'
    return '_AU' + labels['AU'] + '_EX' + labels['EX'] + '_VA' + labels['VA']
