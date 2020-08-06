#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from time import time

from conll_text_manip import load_text_data, save_embeds


parser = argparse.ArgumentParser()
parser.add_argument('data_file', help="Fichier de données d'apprentissage", type=str)
parser.add_argument('window', help="Contexte window size", type=int)
parser.add_argument('embeds_dim', help="Embeddings dimension", type=int)
parser.add_argument('--disable_cuda', action='store_true', help="Disable CUDA")

args = parser.parse_args()
t0 = time()
print("Lançant CBOW...")
data = load_text_data(args.data_file)[:161]

torch.manual_seed(1)

args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print("Device :", args.device)

CONTEXT_SIZE = args.window
EMBEDDING_DIM = args.embeds_dim

tok_sents = []

vocab = set(["*d"+str(i)+"*" for i in range(CONTEXT_SIZE)] + ["*f"+str(i)+"*" for i in range(CONTEXT_SIZE)])

for sent in data:
    tok_sent = sent.split()
    vocab = vocab.union(set(tok_sent))
    tok_sents.append(tok_sent)

vocab_size = len(vocab)

print("Vocabulaire de taille : %d appris (%.3f sec)" % (vocab_size, time()-t0))

i2w = list(vocab)
w2i = {w: i for i, w in enumerate(i2w)}

def create_examples(tok_sents, w2i):
    examples = []
    for sent in tok_sents:
        sent = ["*d"+str(i)+"*" for i in range(CONTEXT_SIZE)] + sent + ["*f"+str(i)+"*" for i in range(CONTEXT_SIZE)]
        sent = [ w2i[t] for t in sent ]
        for i in range(2, len(sent) - 2):
            context = [ sent[i-2], sent[i-1], sent[i+1], sent[i+2] ]
            target = sent[i]
            examples.append((context, target))
    return examples

examples = create_examples(tok_sents, w2i)

class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        batch_size = embeds.size()[0]
        sum_embeds = torch.sum(embeds, dim=1)

        out = self.linear(sum_embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = CBOW(len(vocab), EMBEDDING_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.00001)

BATCH_SIZE = 20
from random import shuffle
t0 = time()
print("Phase d'apprentissage en cours...")
for epoch in range(100):
    total_loss = 0
    shuffle(examples)
    i = 0
    #@@ optimisation stochastique en utilisant un minibatch
    while i < len(examples):
        batch = examples[i: i+BATCH_SIZE]
        (contexts, labels) = zip(*examples)

        # Step 1. Prepare the inputs to be passed to the model
        batch_inputs = torch.tensor(contexts, dtype=torch.long)
        batch_targets = torch.tensor(labels, dtype=torch.long)

        # Step 2. Cleaning gradients
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(batch_inputs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # words wrapped in a tensor)
        loss = loss_function(log_probs, batch_targets)

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        i += 1
    print(str(epoch+1)+"% loss = "+str(total_loss))
    losses.append(total_loss)
t1 = time() - t0
print("Phase d'apprentissage accomplie pour %d dimensions d'embeddings et une fenêtre contextuelle de taille %d (%d min)" % (EMBEDDING_DIM, CONTEXT_SIZE, ((t1%60)*60)/(t1*100)))
trained_weight_context = model.embeddings.weight.data.numpy()
trained_weight_target = model.linear.weight.data.numpy()

save_embeds(i2w, trained_weight_target, "cbow_c"+str(CONTEXT_SIZE)+"_d"+str(EMBEDDING_DIM)+"_target.emb")
save_embeds(i2w, trained_weight_target, "cbow_c"+str(CONTEXT_SIZE)+"_d"+str(EMBEDDING_DIM)+"_context.emb")
print("Fichiers enregistrés.")
