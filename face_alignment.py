"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import cv2 as cv
import numpy as np
import os

def align_rescale_face(image, M):
    aligned = cv.warpAffine(image, M, (112, 112), flags=cv.INTER_CUBIC, borderValue=0.0)
    return aligned

def render_img_and_mask(img, mask, frame_nr, render_path, mask_path):
    frame_nr_str = str(frame_nr).zfill(5)
    frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    output_filepath = os.path.join(render_path, frame_nr_str + '.jpg')
    cv.imwrite(output_filepath, frame, [int(cv.IMWRITE_JPEG_QUALITY), 95])
    frame_mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    output_filepath = os.path.join(mask_path, frame_nr_str + '.jpg')
    cv.imwrite(output_filepath, frame_mask, [int(cv.IMWRITE_JPEG_QUALITY), 100])

def draw_mask(points, image):
    line_type = cv.LINE_8
    left_eyebrow = points[17:22, :]
    right_eyebrow = points[22:27, :]
    nose_bridge = points[28:31, :]
    chin = points[6:11, :]
    mouth_outer = points[48:60, :]
    left_eye = points[36:42, :]
    right_eye = points[42:48, :]
    pts = [np.rint(mouth_outer).reshape(-1, 1, 2).astype(np.int32)]
    cv.polylines(image, pts, True, color=(255, 255, 255), thickness=1, lineType=line_type)
    pts = [np.rint(left_eyebrow).reshape(-1, 1, 2).astype(np.int32)]
    cv.polylines(image, pts, False, color=(223, 223, 223), thickness=1, lineType=line_type)
    pts = [np.rint(right_eyebrow).reshape(-1, 1, 2).astype(np.int32)]
    cv.polylines(image, pts, False, color=(191, 191, 191), thickness=1, lineType=line_type)
    pts = [np.rint(left_eye).reshape(-1, 1, 2).astype(np.int32)]
    cv.polylines(image, pts, True, color=(159, 159, 159), thickness=1, lineType=line_type)
    pts = [np.rint(right_eye).reshape(-1, 1, 2).astype(np.int32)]
    cv.polylines(image, pts, True, color=(127, 127, 127), thickness=1, lineType=line_type)
    pts = [np.rint(nose_bridge).reshape(-1, 1, 2).astype(np.int32)]
    cv.polylines(image, pts, False, color=(63, 63, 63), thickness=1, lineType=line_type)
    pts = [np.rint(chin).reshape(-1, 1, 2).astype(np.int32)]
    cv.polylines(image, pts, False, color=(31, 31, 31), thickness=1, lineType=line_type)
