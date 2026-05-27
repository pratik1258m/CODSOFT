"""
Image Captioning AI Model Architecture
CODSOFT AI Internship - Task 3

This script implements a classic deep learning architecture for image captioning:
1. EncoderCNN: A pre-trained CNN (ResNet-50) used to extract high-level feature vectors from input images.
2. DecoderRNN: A recurrent network (LSTM) that takes the feature vector and sequence tokens to generate descriptions.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        # Load a pre-trained ResNet-50 model
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove the final classification layer (fc layer)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Linear layer to project CNN features to embedding size
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Freeze or unfreeze CNN weights
        for param in self.resnet.parameters():
            param.requires_grad = train_CNN

    def forward(self, images):
        # Extract features of shape (batch_size, 2048, 1, 1)
        features = self.resnet(images)
        # Flatten features to shape (batch_size, 2048)
        features = features.view(features.size(0), -1)
        # Project to embedding size (batch_size, embed_size)
        features = self.dropout(self.relu(self.embed(features)))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # Word embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM Decoder
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Fully connected layer to map LSTM hidden state to word probability distribution
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # captions shape: (batch_size, sequence_length - 1)
        # Remove the <end> token from the captions input to maintain alignment
        embeddings = self.dropout(self.embed(captions[:, :-1]))
        
        # Concatenate the visual features (acting as the <start> token representation)
        # features shape: (batch_size, embed_size) -> unsqueeze to (batch_size, 1, embed_size)
        # embeddings shape: (batch_size, sequence_length - 1, embed_size)
        # Combined inputs shape: (batch_size, sequence_length, embed_size)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # Pass through LSTM
        hiddens, _ = self.lstm(inputs)
        
        # Project to vocabulary space
        outputs = self.linear(hiddens)
        return outputs


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, train_CNN=False):
        super(ImageCaptioningModel, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size, train_CNN)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def generate_caption(self, image, max_length=20, vocab=None):
        """
        Greedy search generation of a caption for a single image during inference.
        """
        result_caption = []
        with torch.no_grad():
            # Extract image features
            features = self.encoderCNN(image).unsqueeze(0)  # Shape: (1, 1, embed_size)
            states = None
            
            # Start token index (usually 1, let's assume standard indexes)
            # In a typical student project, <start> is 1 and <end> is 2.
            input_word = torch.tensor([1]).unsqueeze(0)  # Shape: (1, 1)
            
            # Feed visual feature vector as first input to LSTM
            lstm_input = features
            
            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(lstm_input, states)
                outputs = self.decoderRNN.linear(hiddens.squeeze(1))
                predicted = outputs.argmax(1)
                
                word_idx = predicted.item()
                
                # Check for end of sentence token
                if word_idx == 2:  # Assume index 2 is <end>
                    break
                
                if vocab:
                    word = vocab.get_word(word_idx)
                    result_caption.append(word)
                else:
                    result_caption.append(str(word_idx))
                
                # Embed the predicted word for the next input step
                lstm_input = self.decoderRNN.embed(predicted).unsqueeze(0)
                
        return result_caption


# Simple vocabulary helper to demonstrate academic training setup
class Vocabulary:
    def __init__(self):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.idx = 4

    def add_word(self, word):
        if word not in self.stoi:
            self.stoi[word] = self.idx
            self.itos[self.idx] = word
            self.idx += 1

    def get_word(self, index):
        return self.itos.get(index, "<unk>")

    def __len__(self):
        return len(self.itos)


if __name__ == "__main__":
    # Quick instantiation test to verify imports and tensor shapes
    print("Initializing ResNet50 + LSTM Image Captioning Model...")
    vocab = Vocabulary()
    vocab.add_word("a")
    vocab.add_word("dog")
    vocab.add_word("running")
    vocab.add_word("on")
    vocab.add_word("grass")
    
    model = ImageCaptioningModel(
        embed_size=256,
        hidden_size=256,
        vocab_size=len(vocab),
        num_layers=1
    )
    
    print(f"Model successfully loaded.")
    print(f"Vocabulary Size: {len(vocab)} words.")
    
    # Mock image: shape (batch_size, channels, height, width) -> (1, 3, 224, 224)
    mock_image = torch.randn(1, 3, 224, 224)
    # Mock caption indices: "<start> a dog running <end>" -> [1, 4, 5, 6, 2]
    mock_caption = torch.tensor([[1, 4, 5, 6, 2]])
    
    # Forward pass test
    outputs = model(mock_image, mock_caption)
    print(f"Output predictions tensor shape: {outputs.shape} (batch, sequence_length, vocab_size)")
