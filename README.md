# Vascular Tree Lifting - Group 5 - CMPT 340 D100

Group 5 Members:
- Jimson Zheng 301364719 (jimsonz@sfu.ca)
- Lydia Kim 301415914 (ska238@sfu.ca)
- Jasmine Rai 301347393 (jra58@sfu.ca)
- Kingsley Umeh 301401702 (kumeh@sfu.ca)
- Kevin Beja 301101226 (kdb4@sfu.ca)

## Data

The VascuSynth dataset is located in the 'vascular_tree_data/' folder. 

Source: https://vascusynth.cs.sfu.ca/Data.html

## Usage

### Prerequisites

- Python3
- CPU or NVIDIA GPU

### Installation

Clone this repo:

```
git clone https://github.com/lydiakim97/CMPT340-Vascular-Tree-Lifting.git
cd CMPT340-Vascular-Tree-Lifting
```

In order to install the required python packages run (in a new virtual environment):

```
pip install -r requirements.txt
```

### Training the Model

For training, execute the following command:

`python train.py`

Currently, the model is saved after every 10 epochs.


### Testing the Model

Ensure the EncoderPath and GeneratorPath to the saved model are correct. This will be used to load the trained model.

For testing the model, execute the following command:

`python test.py`


### Visualizing Results
3D visualizations are saved in the 'output/progress' directory. 

The MATLAB script, `createGif.m`, can be used to visualize progress of the model across epochs.


## Sources

- VascuSynth data: https://vascusynth.cs.sfu.ca/Data.html
```
@ARTICLE{cmig2010,
   AUTHOR       = {Ghassan Hamarneh and Preet Jassi},
   JOURNAL      = {Computerized Medical Imaging and Graphics},
   TITLE        = {VascuSynth: Simulating Vascular Trees for Generating
                   Volumetric Image data with Ground Truth Segmentation and
                   Tree Analysis},
   YEAR         = {2010},
   NUMBER       = {8},
   PAGES        = {605-616},
   VOLUME       = {34},
   PDF          = {http://www.cs.sfu.ca/~hamarneh/ecopy/cmig2010.pdf},
   DOI          = {10.1016/j.compmedimag.2010.06.002}
}

@ARTICLE{ij2011,
   AUTHOR       = {Preet Jassi and Ghassan Hamarneh},
   JOURNAL      = {Insight Journal},
   TITLE        = {VascuSynth: Vascular Tree Synthesis Software},
   YEAR         = {2011},
   PAGES        = {1-12},
   VOLUME       = {January-June},
   PDF          = {http://www.cs.sfu.ca/~hamarneh/ecopy/ij2011.pdf},
   DOI          = {10380/3260}
}
```

- PlatonicGAN
GitHub repository: https://github.com/henzler/platonicgan 

[Escaping Plato’s Cave: 3D Shape from Adversarial Rendering](https://geometry.cs.ucl.ac.uk/projects/2019/platonicgan/paper_docs/platonicgan.pdf) (ICCV2019). 

More detailed information corresponding to this paper and code can be found on the following [project page](https://geometry.cs.ucl.ac.uk/projects/2019/platonicgan/).

```
@InProceedings{henzler2019platonicgan,
author = {Henzler, Philipp and Mitra, Niloy J. and Ritschel, Tobias},
title = {Escaping Plato's Cave: 3D Shape From Adversarial Rendering},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```

- Loss functions:  https://github.com/JunMa11/SegLoss

- https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py

- IoU loss: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch

