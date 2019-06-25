import pandas as pd
import numpy as np
import re
import string
import random
import operator
import torch
from torch.utils.data import Dataset, DataLoader


output_file = 'results2.txt'
num_previous_epochs = 250


music = pd.read_csv('lyrics.csv')

hh_lyrics  = music[music['genre'] == 'Hip-Hop']['lyrics']
hh_lyrics_ = [h for h in hh_lyrics if type(h) == str]
hh_lyrics_ = [h.lower() for h in hh_lyrics_]
hh_lyrics_ = [h.translate(str.maketrans('', '', string.punctuation)) for h in hh_lyrics_]
hh_lyrics_ = [h.split() for h in hh_lyrics_]

# count up the words
counter = {}
word2idx = {}

word2idx['<UNK>'] = 0
word2idx['<EOS>'] = 1

for song in hh_lyrics_:
    for word in song:
        if word in counter.keys():
            counter[word] += 1
        else:
            counter[word] = 1

for k, v in counter.items():
    if v >= 10:
        word2idx[k] = len(word2idx)

V = len(word2idx)

songs = [[word2idx.get(w, 0) for w in song] + [1] for song in hh_lyrics_]


# sample the data
num_samples_per_song = 400
sample_size = 5
song_set = []

for song in songs:
    num_words = len(song)
    if num_words <= sample_size+1:
        break
    for _ in range(num_samples_per_song):
        start = np.random.randint(0, num_words-sample_size-1)
        song_set.append(song[start:start+sample_size+1])

# split the data
num_obs = len(song_set)
train, test = [], []

test_ids = random.sample(range(num_obs), num_obs // 3)
train_ids = list(set(range(num_obs)) - set(test_ids))

for idx in train_ids:
    train.append(song_set[idx])

for idx in test_ids:
    test.append(song_set[idx])


class Songs(Dataset):
    def __init__(self, songs):
        self.x = torch.tensor([s[:sample_size] for s in songs])
        self.y = torch.tensor([s[-1] for s in songs]).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


train_ds = Songs(train)
test_ds  = Songs(test)

train_dl = DataLoader(train_ds, 64, shuffle=True)
test_dl  = DataLoader(test_ds, 64)


class LangModel(torch.nn.Module):
    def __init__(self, vocab_size=V, emb_size=30, dropout=0.9, hidden_size=10):
        super(LangModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding   = torch.nn.Embedding(vocab_size, embedding_dim=emb_size)
        self.LSTM = torch.nn.LSTM(input_size=emb_size, hidden_size=hidden_size,
                                  batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size, 2*hidden_size)
        self.fc2 = torch.nn.Linear(2*hidden_size, vocab_size)
        self.fc3 = torch.nn.Linear(hidden_size, vocab_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # we want the final hidden layer, so we will take the
        # hidden part of the h_c output of the LSTM
        batch_size = x.shape[0]
        x = self.embedding(x)
        x, h_c = self.LSTM(x)
        x = self.dropout(h_c[0])
        x = x.squeeze()
        x = self.fc3(x)
        return x.squeeze().double()


def train_model(mod, n_epochs, train_loss=[], test_loss=[]):
    for i in range(num_previous_epochs, num_previous_epochs+n_epochs):
        print('training')
        # train
        total_loss, trials = 0, 0
        mod.train()
        for x, y in train_dl:
            y_hat = mod(x)
            optimizer.zero_grad()

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            trials += y.shape[0]
            total_loss += loss.item()
        print(total_loss/trials)
        with open(output_file, 'a') as f:
            f.write(f'train,{i+1},{total_loss/trials}\n')


        # test
        if i % 5 == 4:
            total_test_loss, test_trials = 0, 0
            mod.eval()
            for x, y in test_dl:
                test_trials += y.shape[0]
                y_hat = mod(x)
                loss = criterion(y_hat, y)
                total_test_loss += loss.item()
            with open(output_file, 'a') as f:
                f.write(f'test,{i+1},{total_test_loss/test_trials}\n')
    return train_loss, test_loss


model = LangModel()
model.load_state_dict(torch.load('./MODEL2.pth'))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_model(model, 100)
torch.save(model.state_dict(), 'MODEL2.pth')
