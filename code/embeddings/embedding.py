import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoModelForMaskedLM, AutoTokenizer, RobertaModel, RobertaTokenizer


class Chemberta_embedding:
    def __init__(self, reduction_method=None):
        self.model = AutoModelForMaskedLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k", output_attentions=True, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")

    def create_embedding(self, smiles):
        input = self.tokenizer.encode(smiles, return_tensors='pt')
        output = self.model(input).hidden_states[-1]
        embedding = output[0,:,:].detach()
        return embedding
        
class Compound:
    def __init__(self, name, smiles=None, chemberta_embedding=None):
        self.name = name
        self.smiles = smiles
        self.chemberta_embedding = chemberta_embedding
        self.chemformers_embedding = None

    
