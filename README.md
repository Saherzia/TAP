# TAP

Installation

This code needs Python-3.7 or higher.

pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip3 install -r requirements.txt

Dataset Preprocessing

Preprocess all datasets

python3 preprocess.py name of dataset

Result 

To run a model on a dataset, run the following command:

python3 main.py --model <TAP> --dataset <dataset> --retrain
