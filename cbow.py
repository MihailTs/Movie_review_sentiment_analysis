import json
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# continuous bag of words
class CBOWDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        context, target = self.data[index]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)
    
df_reviews = pd.read_csv("movies_cleaned.csv")
corpus = df_reviews["text"]
# contextless padding
PADDING = "<PAD>"
UNKNOWN = "<UNK>"

corpus_words = [word for sentence in corpus for word in sentence.split()]

word_counts = Counter(corpus_words)
# keep only words that appear >= 3 times
vocabulary = [word for word, count in word_counts.items() if count >= 3]
vocabulary.append(PADDING)
vocabulary.append(UNKNOWN)
word2idx = {word: i for i, word in enumerate(vocabulary)}
idx2word = {i: word for word, i in word2idx.items()}
vocabulary_size = len(vocabulary)

# actually half the size of the context window
window_size = 2

# generates touples (context, target) - for example:
# the cat chased the bird -> ([the, cat, the, bird], chased); ([the, chased, the], cat)
def generate_corpus_data(corpus, window_size):
    data = []

    for sentence in corpus:
        words = [PADDING] * window_size
        words += sentence.split()
        words += [PADDING] * window_size
        
        for i in range(window_size, len(words) - window_size):
            context = []
            
            for j in range(-window_size, window_size + 1):
                if j != 0:
                    word = words[i + j]
                    context.append(word2idx.get(word, word2idx[UNKNOWN]))
            
            target = word2idx.get(words[i], word2idx[UNKNOWN])
            data.append((context, target))

    return data

corpus_data = generate_corpus_data(corpus, window_size)

dataset = CBOWDataset(corpus_data)    
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    # context shape: (batch_size, 2 * window_size)
    def forward(self, context):
        # embeddings shape: (batch_size, context_size, embedding_dim)
        embeddings = self.embedding(context)
        
        # mean_embeds shape: (batch_size, embedding_dim)
        mean_embeds = embeddings.mean(dim=1)

        # output shape: (batch_size, vocab_size)
        output = self.linear(mean_embeds)
        
        return output
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training the model
embedding_dim = 100
model = CBOWModel(vocabulary_size, embedding_dim)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    total_loss = 0
    
    for context, target in dataloader:
        context, target = context.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(context)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# saving the computed embeddings
embeddings = model.embedding.weight.data.cpu().numpy()
np.save("cbow_embeddings.npy", embeddings)
with open("word2idx.json", "w") as f:
    json.dump(word2idx, f)