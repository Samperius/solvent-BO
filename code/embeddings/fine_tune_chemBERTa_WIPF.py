from fine_tuning_classes import smilesDataset, train
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer, RobertaModel, RobertaTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import get_scheduler

import torch
from torch import tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
import sys

#Args training
batch_size = 8
num_epochs = 3
_lr_rate = 0.001
model_path = 'finetuned_chemBERTa'
#Args data
max_smiles_len = 512

model = AutoModelForSequenceClassification.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k", output_attentions=True, output_hidden_states=True, num_labels = 1)
tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
tokenizer.model_max_length = max_smiles_len
print(tokenizer.vocab_size)
print(model.config.vocab_size)
#load wipf dataframe
df = pd.read_excel('data/wipf_1.7.xlsx')

#prepare datasets and loaders
mask = df['has_yield_and_rxn_solvent_smiles']== True
train_df, test_df = train_test_split(df[mask])
train_dataset = smilesDataset(train_df, tokenizer)
eval_dataset = smilesDataset(test_df, tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)


#train
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=_lr_rate)
model, train_losses  = train(model=model, 
                             train_dataloader=train_dataloader,
                             criterion=criterion, 
                             optimizer=optimizer, 
                             num_epochs=num_epochs, 
                             device=device,
                             model_path = model_path,
                             load=False)


