# Music Fusion
Neural network to generate a new sound timbre from two different instruments.
Network modified based on WaveNet.

![](https://i.imgur.com/8YPUbT3.png)

## Setup
The code was implemented in Python 3.9. Install the required dependencies with:
```
pip install -r requirements.txt
```

The wav files for training and inference need to be placed in the `dataset` folder. The `dataset` folder should have the following structure:
```
dataset
├── train
│   ├── flute
│   │   ├── 2202.wav
│   │   ├── 2203.wav
│   │   └── 2204.wav
│   └── violin
│       ├── 2241.wav
│       ├── 2242.wav
│       ├── 2243.wav
│       ├── 2244.wav
│       ├── 2288.wav
│       └── 2289.wav
└── val
    ├── flute
    │   └── sample_flute.wav
    └── violin
        └── sample_violin.wav
```

## Training
To train the model, run:
```
python train.py
```
Hyperparameters can be modified in `hyp.py`.

## Inference
To generate a new sound timbre, run:
```
python inference.py --model <model_path>
```
