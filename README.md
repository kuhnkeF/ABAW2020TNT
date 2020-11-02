# TSAV Affect Analysis in the Wild (ABAW2020 submission)

**[Two-Stream Aural-Visual Affect Analysis in the Wild](https://arxiv.org/pdf/2002.03399.pdf)**

*(Submission to the Affective Behavior Analysis in-the-wild ([ABAW](https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/)) 2020 competition)*


## Getting Started

Required packages:

PyTorch 1.4, Torchaudio 0.4.0, tqdm, Numpy, OpenCV 4.2.0

You also need

mkvmerge, mkvextract (from mkvtoolnix)

ffmpeg

sox

## Testing

UPDATE 04.06.2020: Please delete dataset.pkl from your database folder and run the updated code. This will improve performance.

To reproduce the competition results, download our model and alignment files:  
[Model](https://www.tnt.uni-hannover.de/project/affwild2/aff2model_tnt.zip) and
[Alignment data](https://www.tnt.uni-hannover.de/project/affwild2/aff2alignmentdata_tnt.zip)

Note: The model is submission 1 from the ABAW competition
We also provide the model for submission 4 with higher performance.
[Submission 4 model](https://www.tnt.uni-hannover.de/project/affwild2/aff2model_tntsub4.zip)

(Alignmentdata file was reuploaded on 11.05.2020). md5sum aff2alignmentdata_tnt.zip 806aee937c52103ad7f48b1b19608860)

You need the original videos from [ABAW](https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/).

Clone the repository and extract the data before running create_database.py. 

create_database.py extracts and aligns the faces and audio files from the Aff-Wild2 videos.

test_val_aff2.py produces the val and test label files.

Please make sure to check the paths in both files. 

Be aware, the whole process takes long time and some disk space. 

(Database creation: face extraction, face-alignment, mask rendering, audio extraction, 3 hours+ and about 31 GiB)
(Model inference: 7 hours on RTX 2080 Ti)

## Citation

Please cite our paper in your publications if the paper/our code or our database alignment/mask data helps your research:

    @INPROCEEDINGS {,
    author = {F. Kuhnke and L. Rumberg and J. Ostermann},
    booktitle = {2020 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2020) (FG)},
    title = {Two-Stream Aural-Visual Affect Analysis in the Wild},
    year = {2020},
    volume = {},
    issn = {},
    pages = {366-371},
    keywords = {expression recognition;action units;affective behavior analysis;human computer interaction;valence arousal;emotion recognition},
    doi = {10.1109/FG47880.2020.00056},
    url = {https://doi.ieeecomputersociety.org/10.1109/FG47880.2020.00056},
    publisher = {IEEE Computer Society},
    address = {Los Alamitos, CA, USA},
    month = {may}
    }


Link to the paper:

[IEEE TSAV](https://www.computer.org/csdl/proceedings-article/fg/2020/307900a366/1kecIcAu7W8)

[Arxiv TSAV](https://arxiv.org/pdf/2002.03399.pdf)

Model and alignment data is restricted for research purposes only.
By using the dataset, code or alignments, please acknowledge the effort by citing the corresponding papers.
