#Preparing dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm


class smilesDataset(Dataset):
        def __init__(self, df, tokenizer) -> None:
                super().__init__()
                self.smiles = df['rxn_smiles_with_solvent']
                self.labels = df['product_yield']
                self.tokenizer = tokenizer
        
        def __len__(self):
                return self.smiles.shape[0]
        
        def __getitem__(self, index) -> None:
                return torch.tensor(self.tokenizer.encode(self.smiles.iloc[index], padding="max_length", max_length=512, truncation= True)), torch.tensor(self.labels.iloc[index])



def train(train_dataloader, model, criterion, optimizer, num_epochs, device, model_path, load=False):
    
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    losses = []
    model.train()
    if load:
        model.load_state_dict(torch.load(model_path))
    
    for epoch in range(num_epochs):
        epoch_cum_loss = 0
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            X, y = batch[0].to(device), batch[1].to(device)
            attention_mask = X>1
            pred = torch.sigmoid(model(X, attention_mask=attention_mask).logits)
            pred = pred.reshape(y.shape)
            loss = criterion(pred.float(), y.float())
            epoch_cum_loss += loss
            losses.append(loss)
            print(f'epoch {epoch+1}/{num_epochs}, batch number {i+1}/{len(train_dataloader)}, batch cumloss {epoch_cum_loss/(i+1)}')
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
        torch.save(model.state_dict(), model_path)
    return model, losses