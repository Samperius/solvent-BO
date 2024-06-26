import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dimension,reduced_dimension):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, reduced_dimension)  # Reducing to 2 dimensions
        )
        self.decoder = nn.Sequential(
            nn.Linear(reduced_dimension, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dimension),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Dimensionality_reduction_with_AE:
    def __init__(self, input_dimension, reduced_dimension):
        self.model = Autoencoder(input_dimension, reduced_dimension)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self, data, epochs):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in data:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(output, batch)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(data)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}')
    
    def reduce_dimension(self, data):
        return self.model.encoder(data)