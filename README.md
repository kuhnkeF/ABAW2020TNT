# TSAV Affect Analysis in the Wild (ABAW2020 submission)

**[Two-Stream Aural-Visual Affect Analysis in the Wild](TBA)**
*(Submission to the Affective Behavior Analysis in-the-wild (ABAW) 2020 competition)*


## Getting Started

Required packages:

PyTorch 1.4, Torchaudio 0.4.0, tqdm, Numpy, OpenCV 4.2.0

You also need

mkvmerge, mkvextract (from mkvtoolnix)

ffmpeg

sox

## Testing

To reproduce the competition results, download our model and alignment files:  
[Model](https://www.tnt.uni-hannover.de/project/affwild2/aff2model_tnt.zip) and
[Alignment data](https://www.tnt.uni-hannover.de/project/affwild2/aff2alignmentdata_tnt.zip)

Clone the repository and extract the data before running create_database.py. 

create_database.py extracts and aligns the faces and audio files from the Aff-Wild2 videos.

test_val_aff2.py produces the val and test label files.

Please make sure to check the paths in both files. 

Be aware, the whole process takes long time and some disk space. 

(Database creation: face extraction, face-alignment, mask rendering, audio extraction, 3 hours+ and about 31 GiB)
(Model inference: 7 hours on RTX 2080 Ti)

## Citation

Please cite our paper in your publications if the paper/our code or our database alignment/mask data helps your research:

*we will update the reference as soon as the paper is published*

    @inproceedings {KuhnkeRumberg2020TSAV,
      author = {Felix Kuhnke and Lars Rumberg and J{\"o}rn Ostermann},
      title = {Two-Stream Aural-Visual Affect Analysis in the Wild},
      booktitle = {TBA},
      year = {2020},
      month = TBA
    }

Link to the paper:

*we will provide a link to the paper as soon as it is published*

Model and alignment data is restricted for research purposes only.
By using the dataset, code or alignments, please acknowledge the effort by citing the corresponding papers.