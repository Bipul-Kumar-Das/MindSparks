from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd
import os
import re
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from nltk.tokenize import word_tokenize
import torch
import nltk
from torchvision.models import ResNet50_Weights
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.metrics import edit_distance
nltk.download('punkt_tab')  #for .py file
#nltk.download('punkt')      #for .ipynb file
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

train_images_input_folder = 'all_data/train/radiology/images'
train_images_captions = 'all_data/train/radiology/captions.txt'
train_images_caption_mappings = 'all_data/train/radiology/traindata.csv'

if os.path.exists(train_images_captions):
    with open(train_images_captions, 'r', encoding="utf-8") as caption:
        all_captions = caption.readlines()
else:
    print("File not found!")
print('Number of data instance available in train data is : ', len(all_captions))

data = pd.read_csv(train_images_caption_mappings)
data['name'] = data['name'].str.strip()
#print(data.columns)
#print(data.head)

train_image_ids = []
train_images = []
train_captions = []
image_counter = 0

for sentence in all_captions:
    sentence = sentence.strip()
    sentence_list = re.split(r'[ \t]+', sentence)
    image_id = sentence_list[0]
    caption = ' '.join(sentence_list[1:])

    # Get the image file name using .iloc[0]['name'] for the first matching row
    image_file_name = data.loc[data['id'] == image_id, 'name'].iloc[0]

    # Remove any leading/trailing whitespace and newline characters from image_file_name
    image_file_name = image_file_name.strip()

    #print(image_id, type(image_id))
    #print(image_file_name, type(image_file_name))
    #print(caption, type(caption))

    image_path = os.path.join(train_images_input_folder, image_file_name)
    try:
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        image_counter += 1
        #print(f"{image_counter}. {image_id}")

        train_image_ids.append(image_id)
        train_images.append(image_array)
        train_captions.append(caption)

        #if(image_counter==80):   #REMOVE 1 FOR 30%
        if(image_counter==int(0.3 * len(all_captions))):  # To take 30% of train data
            break

    except UnidentifiedImageError:  # Skip if image can't be opened
        print(f"Warning: Skipping {image_file_name} due to UnidentifiedImageError. It might be corrupted or in an unsupported format.")
        #break

    except Exception as e:  # Catch other potential errors
        print(f"Warning: An error occurred while processing {image_file_name}: {e}")
        #break

#print('\nTotal number of train image_id is : ', len(train_image_ids), type(train_image_ids[0]))
print('\nTotal number of train images is : ', len(train_images), type(train_images[0]))
#print('\nTotal number of train captions is : ', len(train_captions), type(train_captions[0]))


test_images_input_folder = 'all_data/test/radiology/images'
test_images_captions = 'all_data/test/radiology/captions.txt'
test_images_caption_mappings = 'all_data/test/radiology/testdata.csv'

if os.path.exists(test_images_captions):
    with open(test_images_captions, 'r', encoding="utf-8") as caption:
        all_captions = caption.readlines()
else:
    print("File not found!")
print('Number of data instance available in test data is : ', len(all_captions))

data = pd.read_csv(test_images_caption_mappings)
data['name'] = data['name'].str.strip()
#print(data.columns)
#print(data.head)

test_image_ids = []
test_images = []
test_captions = []
image_counter = 0

for sentence in all_captions:
    sentence = sentence.strip()
    sentence_list = re.split(r'[ \t]+', sentence)
    image_id = sentence_list[0]
    caption = ' '.join(sentence_list[1:])

    # Get the image file name using .iloc[0]['name'] for the first matching row
    image_file_name = data.loc[data['id'] == image_id, 'name'].iloc[0]

    # Remove any leading/trailing whitespace and newline characters from image_file_name
    image_file_name = image_file_name.strip()

    #print(image_id, type(image_id))
    #print(image_file_name, type(image_file_name))
    #print(caption, type(caption))

    image_path = os.path.join(test_images_input_folder, image_file_name)
    try:
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        image_counter += 1
        #print(f"{image_counter}. {image_id}")

        test_image_ids.append(image_id)
        test_images.append(image_array)
        test_captions.append(caption)

        #if(image_counter==8):  #REMOVE 2 FOR 30%  
        if(image_counter==int(0.3 * len(all_captions))):  # To take 30% of train data
            break

    except UnidentifiedImageError:  # Skip if image can't be opened
        print(f"Warning: Skipping {image_file_name} due to UnidentifiedImageError. It might be corrupted or in an unsupported format.")
        #break

    except Exception as e:  # Catch other potential errors
        print(f"Warning: An error occurred while processing {image_file_name}: {e}")
        #break

#print('\ntotal number of test image_id is : ', len(test_image_ids), type(test_image_ids[0]))
print('\ntotal number of test images is : ', len(test_images), type(test_images[0]))
#print('\ntotal number of test captions is : ', len(test_captions), type(test_captions[0]))



# Define a simple vocabulary and sample dataset setup
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<MASK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<MASK>"}
        self.idx = 4

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx["<PAD>"])

    def __len__(self):
        return len(self.word2idx)

# Build vocabulary from captions
def build_vocab(captions):
    vocab = Vocabulary()
    for caption in captions:
        for word in word_tokenize(caption.lower()):
            vocab.add_word(word)
    return vocab

# Example Dataset Class
class ImageCaptionDataset(Dataset):
    def __init__(self, images, captions, vocab, max_caption_length, transform=None):
        self.images = images
        self.captions = [self.preprocess_caption(c, vocab) for c in captions]
        self.vocab = vocab
        self.transform = transform
        self.max_caption_length = max_caption_length

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        caption = self.captions[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(caption, dtype=torch.long)

    def preprocess_caption(self, caption, vocab):
        tokens = word_tokenize(caption.lower())
        number_of_paddings = max_caption_length - len(tokens)
        #indices = [vocab("<START>")] + [vocab(token) for token in tokens] + [vocab("<END>")] + [vocab("<PAD>")] * number_of_paddings
        indices = [vocab("<START>")] + [vocab(token) for token in tokens] + [vocab("<END>")] + [vocab("<PAD>")] * (number_of_paddings+1)
        return indices

#images = train_images
#captions = train_captions
vocab = build_vocab(train_captions)
vocab_size = len(vocab)
print('Size of vocabulary is : ', vocab_size)

# Data Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Caption padding to make same size
max_caption_length = max(len(word_tokenize(seq.lower())) for seq in train_captions)

# DataLoader
dataset = ImageCaptionDataset(train_images, train_captions, vocab, max_caption_length, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
#dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
print('Length of dataloader is : ', len(dataloader))




# Model Definitions
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        #self.cnn = models.resnet50(pretrained=True)
        self.cnn = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)

    def forward(self, images):
        features = self.cnn(images)
        return features

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(CaptionDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        #print('Received features: ', features.shape)
        #print('Received captions: ', captions.shape)

        embedded = self.embedding(captions)
        #print('Embedded captions: ', embedded.shape)

        features = features.unsqueeze(1)  # Add time dimension
        #print('Expanded features: ', features.shape)

        inputs = torch.cat((features, embedded), dim=1)
        #print('Inputs: ', inputs.shape)

        outputs, _ = self.lstm(inputs)
        #print('Outputs: ', outputs.shape)

        predictions = self.fc(outputs)
        #print('Predictions: ', predictions.shape)

        return predictions

# Initialize Models
encoder = ImageEncoder()
decoder = CaptionDecoder(vocab_size=vocab_size, embedding_dim=256, hidden_dim=512)

# Loss and Optimizer
#criterion = nn.CrossEntropyLoss(ignore_index=0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

total_encoder_parameters = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
total_decoder_parameters = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
print('Number of encoder parameters: ', total_encoder_parameters)
print('Number of decoder parameters: ', total_decoder_parameters)

# Inference Function
def predict_caption(encoder, decoder, image, vocab, max_length=max_caption_length):
    features = encoder(image.unsqueeze(0))
    caption = [vocab("<START>")]
    for _ in range(max_length):
        inputs = torch.tensor(caption).unsqueeze(0)
        outputs = decoder(features, inputs)
        _, predicted = outputs[:, -1].max(1)
        word = predicted.item()
        caption.append(word)
        if word == vocab("<END>") or word == vocab("<PAD>"):
            break

    # Convert word indices to words
    generated_caption = [vocab.idx2word[idx] for idx in caption]

    # Remove <START> and <END> tokens for the final output
    generated_caption = generated_caption[1:-1]  # Exclude <START> and <END>
    return generated_caption



def calculate_rouge_scores(reference_caption, generated_caption):

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference_caption, generated_caption)

    # Extract and format scores
    rouge_scores = scores['rougeL'].fmeasure
   
    return rouge_scores

blue_score_per_epoch = []
levenshtein_distance_per_epoch = []
rouge_score_per_epoch = []

# Training Loop
num_epochs = 3
for epoch in range(num_epochs):
    for image, caption in dataloader:
        image, caption = image.float(), caption
        #print("image", image.shape, "caption", caption.shape)

        features = encoder(image)
        #print("features", features.shape)

        outputs = decoder(features, caption[:, :-2])
        #outputs = decoder(features, caption[:, :-1]) #164 sent and from that too 1 <start> tag is removed while calculating loss, so resultant is 163 but 165 received
        #print("outputs", outputs.shape)

        loss = criterion(outputs.view(-1, vocab_size), caption[:, 1:].reshape(-1))
        #loss = criterion(outputs.view(-1, vocab_size), caption.reshape(-1))
        #print("loss", loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    

    #images = test_images
    #captions = test_captions
    bleu_score_sum = 0
    error_sum = 0
    rouge_sum = 0

    for i in range(len(test_images)):
        # Example Inference
        #sample_image = torch.tensor(transform(images[i]), dtype=torch.float)
        sample_image = transform(test_images[i]).clone().detach().float()
        #print(type(sample_image), sample_image.shape)
        caption = predict_caption(encoder, decoder, sample_image, vocab)
        predicted_caption = " ".join(caption)
        #print("Predicted Caption:", predicted_caption)
        #print("Expected Caption:", test_captions[0])

        # Compute Levenshtein Distance
        error = edit_distance(test_captions[i].split(), predicted_caption.split())
        #print("Levenshtein Distance:", error)

        # Compute BLEU Score
        smoothing_function = SmoothingFunction().method4  # You can experiment with different methods
        bleu_score = sentence_bleu(test_captions[i], predicted_caption, smoothing_function=smoothing_function)
        #print('Bleu score is : ', bleu_score)

        rouge_scores = calculate_rouge_scores(test_captions[i], predicted_caption)
        
        bleu_score_sum += bleu_score
        error_sum += error
        rouge_sum += rouge_scores
        
    #print("BLEU score sum : ", bleu_score_sum)
    #print("BLEU Score Average : ", bleu_score_sum/len(test_images))
    blue_score_per_epoch.append(bleu_score_sum/len(test_images))
    
    #print("Levenshtein Distance sum : ", error_sum)
    #print("Levenshtein Distance Average : ", error_sum/len((test_images)))
    levenshtein_distance_per_epoch.append(error_sum/len((test_images)))
    
    rouge_score_per_epoch.append(rouge_sum/len(test_images))
                                   

# Define file paths for saving the models
encoder_path = "encoder"
decoder_path = "decoder"

# Save the state dictionaries
torch.save(encoder.state_dict(), encoder_path)
torch.save(decoder.state_dict(), decoder_path)
print("Models saved successfully!")

print('\nblue_score_per_epoch score for 3 epochs are :', blue_score_per_epoch)
print('\nlevenshtein_distance_per_epoch score for 3 epochs are :', levenshtein_distance_per_epoch)  
print('\nrouge_score_per_epoch score for 3 epochs are :', rouge_score_per_epoch)  



#print(len(test_images))
# Example Inference
sample_image = transform(test_images[4]).clone().detach().float()
#print(type(sample_image), sample_image.shape)

caption = predict_caption(encoder, decoder, sample_image, vocab)
predicted_caption_load = " ".join(caption)
print("Predicted Caption:", predicted_caption_load)
print("Expected Caption:", test_captions[4])

# Convert the tensor to a format suitable for Matplotlib (Channel first to HxWxC)
image = sample_image.numpy()
image = np.transpose(image, (1, 2, 0))

# Plot the image
plt.imshow(image)
plt.axis('off')
plt.show()