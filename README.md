# TAP

** Transformer-Based Adversarial Perturbations Model **  

This repository contains the implementation of a transformer-based model for anomaly detection.  

**Installation**  

Ensure you have **Python 3.7** or higher installed. Then, install the required dependencies:  


pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt


##**Dataset Preprocessing**  

Preprocess the dataset before running the model:  


python3 preprocess.py <dataset_name>


##**Running the Model**  

To train and evaluate the model on a dataset, use the following command:  


python3 main.py --model <model_name> --dataset <dataset_name> --retrain


For more details, check the code and configurations.  
  
