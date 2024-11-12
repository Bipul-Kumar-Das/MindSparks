Downlod the dataset form the link https://www.kaggle.com/datasets/virajbagal/roco-dataset?resource=download-directory
Unzip the zipped dataset and save in the project directory along with the other .py and related requirements files.
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

