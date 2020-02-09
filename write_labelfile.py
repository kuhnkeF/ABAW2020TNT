"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
from utils import *

def au_to_str(arr):
    str = "{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d}".format(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6],
                                                           arr[7])
    return str

def ex_to_str(arr):
    str = "{:d}".format(arr)
    return str

def va_to_str(arr):
    str = "{:.3f},{:.3f}".format(arr[0], arr[1])
    return str

def write_labelfile(label_array, type, video_name, position_str=None, result_dir="", clip=True):
    if type not in ['VA', 'AU', 'EX']:
        raise NameError('unknown type')
    assert isinstance(label_array, np.ndarray)

    # check the size of the labels
    if type == 'VA':
        assert label_array.shape[1] == 2
        if clip:
            label_array = np.clip(label_array, -1.0, 1.0)
        writer = va_to_str

    if type == 'EX':
        if label_array.ndim > 1:
            assert label_array.shape[1] == 1
        if clip:
            label_array = np.clip(label_array, 0.0, 6.0)
        label_array = np.round(label_array).astype(np.int)
        writer = ex_to_str

    if type == 'AU':
        assert label_array.shape[1] == 8
        if clip:
            label_array = np.clip(label_array, 0.0, 1.0)
        label_array = np.round(label_array).astype(np.int)
        writer = au_to_str

    if position_str is None:
        name_pos = video_name
    else:
        name_pos = video_name + position_str
    # Folder
    track_folder ={'AU': 'AU_Set',
                   'VA': 'VA_Set',
                   'EX': 'EXPR_Set'}

    #Header
    header = {"AU": "AU1,AU2,AU4,AU6,AU12,AU15,AU20,AU25", # 0,0,0,0,1,0,0,0
              "VA": "valence,arousal", # 0.602,0.389 or -0.024,0.279
              "EX": "Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise" # 4
             }

    #
    out_dir = os.path.join(result_dir, track_folder[type])
    os.makedirs(out_dir, exist_ok=True)
    label_file_path = os.path.join(out_dir, name_pos + ".txt")
    # remove any old label file
    if os.path.isfile(label_file_path):
        os.remove(label_file_path)
    #write
    text_file = open(label_file_path, "w")
    # write the header
    n = text_file.write(header[type])
    n = text_file.write('\n')
    # write the values
    for i in range(label_array.shape[0]):
        n = text_file.write(writer(label_array[i]))
        n = text_file.write('\n')
    text_file.close()

