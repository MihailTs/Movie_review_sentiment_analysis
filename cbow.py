import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# continuous bag of words
class CBOWDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        context, target = self.data[index]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)
    
# using a simple hardcoded dataset for now
corpus = [
    "the cat sat on the mat",
    "the dog barked at the cat",
    "the bird sang a song",
    "cats and dogs are animals",
    "the mat was sat on by the cat",
    "the cat chased the bird",
    "a bird sang to a bird"
]
# contextless padding
PADDING = "<PAD>"

corpus_words = [word for sentence in corpus for word in sentence.split()]
vocabulary = sorted(set(corpus_words))
vocabulary.append(PADDING)
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
                    context.append(word2idx[words[i + j]])
            
            target = word2idx[words[i]]
            data.append((context, target))

    return data

corpus_data = generate_corpus_data(corpus, window_size)

dataset = CBOWDataset(corpus_data)    
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

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
    

# training the model
embedding_dim = 50
model = CBOWModel(vocabulary_size, embedding_dim)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    total_loss = 0
    
    for context, target in dataloader:
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