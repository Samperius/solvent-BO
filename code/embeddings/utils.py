import pubchempy as pcp
from embedding import Compound

def name_to_smiles(name):
    try:
        return pcp.get_compounds(name, 'name')[0].canonical_smiles
    except:
        print('could not fetch smiles')
        return None

def smiles_to_iupac(smiles, orig_name):
    try:
        compounds = pcp.get_compounds(smiles, namespace='smiles')
        match = compounds[0]
        return match.text.lower()
    except:
        return orig_name

def get_smiles_names_embeddings(emb, list_of_components):
    component_dict = {}
    for i, component_name in enumerate(list_of_components):
        print('component_name:', component_name)
        smiles = name_to_smiles(component_name)
        component_name = smiles_to_iupac(smiles, component_name)
        if smiles is None:
            print('Could not find smiles for', component_name)
            c = Compound(component_name, smiles=None)
            component_dict[component_name] = c
        else:
            c = Compound(component_name, smiles=smiles)
            c.chemberta_embedding = emb.create_embedding(c.smiles)
            component_dict[component_name]= c
    return component_dict

