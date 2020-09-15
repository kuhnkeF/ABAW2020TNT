"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
# this script will create a database folder including the
# cropped and aligned, and mask images
# audio files, and a metadata file
# that will be used to output test and validation results
# please download alignment files and .csv file and
# - set output_dir
# - extract all alignment files to the "output_dir"
# - set csv_file_path
# - set paths to the original aff2 batch1 and batch2 folder

# !!! If you do not have access to the competition set (two files batch1 and batch2) but the normal AFF2
# downloads AUSet, ExpressionSet... check
# -> sort_non_competition_set.py
# and change the video paths and set use_comp_folders to False


from utils import *
import csv
from tqdm import tqdm
import subprocess
from sox import Transformer
from video import Video
from face_alignment import *
import pickle
from sort_non_competition_set import sort_videos


# paths to original aff2 videos
use_comp_folders = True
if use_comp_folders:
    video_paths = ['/data/face/Aff2/tmp/batch1', '/data/face/Aff2/tmp/batch2']
else:
    video_paths = ['/data/face/Aff2/tmp/NON_Competition/ALLVIDEOS'] # just all videos in one folder

csv_file_path = 'video_info.txt'
output_dir = 'aff2_processed/'   # this is where the extracted data will be stored (in /extracted)

def create_database():

    all_videos = []
    for path in video_paths:
        all_videos += find_all_video_files(path)
    all_videos.sort()

    if use_comp_folders is False:
        all_videos = sort_videos(all_videos)

    vid_labels = []
    with open(csv_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            vid_labels.append(row)

    ## GET NEW NAME
    meta_data = {'train_val_test': {}}
    unique_id = 0
    new_vid_names = {}
    for video in all_videos:
        short_vid_name = get_filename(video)
        temp = {}  #
        new_vid_names[short_vid_name] = []

        for label in vid_labels:
            vid_name = label[0]
            label_type = label[1]
            label_in_split = label[2]
            label_position = label[3]
            if vid_name == short_vid_name:  # vid name
                try:
                    if label_type in temp[label_position]:
                        print('Duplicate ' + label_type + ' label file for video: ' + short_vid_name)
                        raise NameError('duplicate')
                    temp[label[3]][label[1]] = {'path': '', 'original_split': label_in_split}
                except:
                    temp[label_position] = {}
                    temp[label_position][label_type] = {'path': '', 'original_split': label_in_split}

        if len(temp) == 0:
            raise NameError('no label')
        else:
            for position in temp:
                unique_id += 1
                if len(temp) == 1 and position == 'main':
                    name = str(unique_id).zfill(3) + get_label_str2(temp[position])
                    new_vid_names[short_vid_name].append(name)
                    meta_data['train_val_test'][name] = {'path':   video,
                                                         'labels': temp[position]}
                else:
                    name = str(unique_id).zfill(3) + get_label_str2(temp[position]) + '_' + position
                    new_vid_names[short_vid_name].append(name)
                    meta_data['train_val_test'][name] = {'path': video, 'labels': temp[position]}

    # create the new files
    os.makedirs(output_dir, exist_ok=True)
    new_vids = meta_data['train_val_test']
    for video in tqdm(new_vids):
        origin = new_vids[video]['path']
        destination = output_dir + '/' + video + os.path.splitext(origin)[1]
        destination_folder = output_dir + '/' + video
        if os.path.isfile(destination):
            os.remove(destination)
        # instead of symlink you could copy and rename the videos
        os.symlink(origin, destination)
        # count the frames and add some infos
        vid = Video(destination)
        vid.meta['original_video'] = get_filename(origin) + get_extension(origin)
        for label_type in new_vids[video]['labels']:
            vid.meta[label_type] = new_vids[video]['labels'][label_type]['original_split']
        vid.write_meta()

def extract_audio():
    all_videos = find_all_video_files(output_dir)
    for video in tqdm(all_videos):
        mkvfile = os.path.join(os.path.dirname(video), 'temp.mkv')
        command = 'mkvmerge -o ' + mkvfile + ' ' + video
        subprocess.call(command, shell=True)
        video_ts_file = os.path.join(os.path.dirname(video), 'video_ts.txt')
        audio_ts_file = os.path.join(os.path.dirname(video), 'audio_ts.txt')
        command = 'mkvextract ' + mkvfile + ' timestamps_v2 0:' + video_ts_file
        subprocess.call(command, shell=True)
        command = 'mkvextract ' + mkvfile + ' timestamps_v2 1:' + audio_ts_file
        subprocess.call(command, shell=True)
        with open(video_ts_file, 'r') as f:
            f.readline()  # skip header
            video_start = f.readline()
        with open(audio_ts_file, 'r') as f:
            f.readline()  # skip header
            audio_start = f.readline()
        offset_ms = int(audio_start) - int(video_start)
        # extract audio
        audio_tmp = os.path.join(os.path.dirname(video), 'temp.wav')
        command = 'ffmpeg -i ' + video + ' -ar 44100 -ac 1 -y ' + audio_tmp
        subprocess.call(command, shell=True)
        # use the offset to pad the audio with zeros, or trim the audio
        audio_name = os.path.splitext(video)[0] + '.wav'
        tfm = Transformer()
        if offset_ms >= 0:
            tfm.pad(start_duration=offset_ms / 1000)
        elif offset_ms < 0:
            tfm.trim(start_time=-offset_ms / 1000)
        tfm.build(audio_tmp, audio_name)
        os.remove(mkvfile)
        os.remove(audio_tmp)
        os.remove(video_ts_file)
        os.remove(audio_ts_file)

def create_alignments():
    # you need to extract all alignment files to the "output_dir"
    out_extracted = os.path.join(output_dir, 'extracted')
    os.makedirs(out_extracted, exist_ok=True)

    all_videos = find_all_video_files(output_dir)
    for video in all_videos:
        video_filename = get_filename(video)
        print(video_filename)
        # load the alignment+mask file
        # create the alignment
        extraction_info_path = os.path.join(output_dir, video_filename + '_alignment_mask.pkl')
        print(extraction_info_path)
        extraction_info = pickle.load(open(extraction_info_path, 'rb'))

        render_path = os.path.join(out_extracted, video_filename)
        mask_path = os.path.join(out_extracted, video_filename, 'mask')
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)

        video = Video(video)
        black_frame = np.zeros((112, 112, 3), np.uint8)
        for i, frame in enumerate(tqdm(video)):
            M = extraction_info['frame'][i]
            # check if there is any data
            if M is None:
                # just render a black frame
                out_frame = black_frame
                mask_frame = black_frame
            else:
                out_frame = align_rescale_face(frame, M)
                mask_frame = black_frame.copy()
                draw_mask(extraction_info['mask'][i],mask_frame)
            render_img_and_mask(out_frame, mask_frame, i, render_path, mask_path)

if __name__ == '__main__':
    print('creating database')
    create_database()
    print('extracting audio')
    extract_audio()
    print('creating face alignments')
    create_alignments()
    print('done')
