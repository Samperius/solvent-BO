# %%
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer
from bertviz import head_view
import torch

# %%
import sys
!test -d bertviz_repo && echo "FYI: bertviz_repo directory already exists, to pull latest version uncomment this line: !rm -r bertviz_repo"
# !rm -r bertviz_repo # Uncomment if you need a clean pull from repo
!test -d bertviz_repo || git clone https://github.com/jessevig/bertviz bertviz_repo
if not 'bertviz_repo' in sys.path:
  sys.path += ['bertviz_repo']
!pip install regex

# %%
!git clone https://github.com/seyonechithrananda/bert-loves-chemistry.git

# %%
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer
from bertviz import head_view

model = AutoModelForMaskedLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")

fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)


# %%
smiles_mask = "C1=CC=CC<mask>C1"
smiles = "C1=CC=CC=C1"

masked_smi = fill_mask(smiles_mask)

for smi in masked_smi:
  print(smi)

# %%
encoded_input = tokenizer.encode(smiles, return_tensors="pt")
output = model(encoded_input)
output

# %%
len(smiles)

# %%
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
cos(torch.randn(1, 128),torch.randn(1, 128)).shape

# %%


# %%
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
cos(output.logits[0][10],output.logits[0][10])

# %%
output.logits[0][10].shape

# %%
ethanol = 'CCO'
ethanol_input = tokenizer.encode(ethanol, return_tensors='pt')
ethanol = model(ethanol_input)

# %%
ethanol.logits

# %%
methanol = 'CO'
methanol_input = tokenizer.encode(methanol, return_tensors='pt')
methanol = model(methanol_input)

# %%
methanol.logits

# %%
test = test.detach()
test.shape

# %%
test.mean(axis=1)

# %%
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
cos(methanol.logits[0].mean(axis=1),ethanol.logits[0].mean(axis=1))

# %%


# %%



