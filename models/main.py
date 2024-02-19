import sys
sys.path.append('/users/Bicaution/')
import numpy as np
import random
import warnings
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm, tqdm_notebook
from models.MYBERT import *
from models.EOS import *
from models.EGCI import *

#construction of the model

class Bicaution(nn.Module):

    def __init__(self, args, event_encoder,hps):
        super(Bicaution, self).__init__()
        self.hps = hps
        self.args = args
        self.event_encoder = event_encoder
        self.reasoner = Reasoning_chain(args)
        self.context_encoder = Context_Encoder(hps)

    def forward(self, input_ids, end_ids, labels, length, sentence_mask=None, attention_mask=None, token_input_ids=None):

        batch_size = input_ids.shape[0]
        max_chain = self.args.max_chain
        max_seq_length = self.max_seq_length
        mlm_probability = self.args.mlm_probability
        contexts = self.context_encoder(input_ids, end_ids, length, sentence_mask)

        input_ids = input_ids.reshape(batch_size * max_chain, -1)
        attention_mask = attention_mask.reshape(batch_size * max_chain, -1)
        token_input_ids = token_input_ids.reshape(batch_size * max_chain, -1)

        sentence_mask = sentence_mask.reshape(batch_size * max_chain, -1)
        end_ids = end_ids.reshape(batch_size * max_chain, -1)

        event_embeddings = \
        self.event_encoder(input_ids, length, attention_mask=attention_mask, token_input_ids=token_input_ids)[0][-1]

        sentence_mask = sentence_mask.unsqueeze(-1)
        sentence_mask = sentence_mask.expand(sentence_mask.shape[0], sentence_mask.shape[1], event_embeddings.shape[-1])

        end_ids = end_ids.unsqueeze(-1)
        end_ids = end_ids.expand(end_ids.shape[0], end_ids.shape[1], event_embeddings.shape[-1])

        event_embeddings = torch.tanh(event_embeddings * sentence_mask)

        event_new_embeddings = event_embeddings[:, ::max_seq_length, :]

        probability_matrix = np.full(labels.shape, mlm_probability)
        masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(np.bool)
        labels[~masked_indices] = -1

        final_scores = self.reasoner(event_new_embeddings, end_ids , contexts, labels)

        return final_scores