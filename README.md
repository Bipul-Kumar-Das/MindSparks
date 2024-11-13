Downlod the dataset form the link https://www.kaggle.com/datasets/virajbagal/roco-dataset?resource=download-directory
Unzip the zipped dataset and save in the project directory along with the other .py and related requirements files.

# Medical Image Captioning using ResNet50 (encoder) and LSTM (decoder) 

Execute the resnet50_lstm.py. This implementation is using 30% of available data for both train & test. Implementation details:

Encoder:`Pretrained ResNet50
	   Output feature vector size = 256
	   Number of parameters = 24032576

Decoder: LSTM
	    Embedding dimension = 256
	    Hidden dimension = 256
	    Number of parameters = 19200902

Vocabulary size = 22918
Criteria = Cross Entropy Loss
Optimizer = Adam

This implementation is giving the performance metrics Blue score, Levenshtein distance and Rouge score over the test set for 3 epochs.


# Medical Image Captioning using Vision-Encoder-Decoder (SWIN + GPT2)

This repository contains code for a medical image captioning system using a Vision-Encoder-Decoder architecture. The system is designed to generate descriptive captions for medical images, particularly focusing on radiology images.

## Requirements

### Hardware Requirements
- CUDA-capable GPU (recommended)
- Minimum 8GB RAM (16GB or more recommended)
- Sufficient storage space for datasets and models

### Software Requirements
```python
pip install transformers
pip install rouge_score
pip install evaluate
pip install datasets
pip install torch
pip install pillow
pip install pandas
pip install tqdm
```

 Setup

```
git clone https://github.com/Bipul-Kumar-Das/MindSparks.git
cd MindSparks
```

Install the required packages:
```
pip install -r requirements.txt
```

Update the data paths in the code to match your local setup
```
train_csv = "path/to/your/traindata.csv"
train_img_folder = "path/to/your/train/images"
val_csv = "path/to/your/valdata.csv"
val_img_folder = "path/to/your/validation/images"
test_csv = "path/to/your/testdata.csv"
test_img_folder = "path/to/your/test/images"
```
