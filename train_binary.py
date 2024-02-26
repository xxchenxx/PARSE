import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import esm
import os

class ProteinDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def load_embeddings_and_labels(sequence_file, label_file, esm_model, esm_alphabet, esm_cache_path='esm_cache', use_cuda=True, repr_layer=12):
    sequences = {}
    with open(sequence_file, 'r') as f:
        current_id = None
        for line in f:
            if line.startswith('>'):
                current_id = line[1:].strip().replace("_", "")
            else:
                seq = line.strip()
                if current_id in sequences:
                    sequences[current_id] += seq
                else:
                    sequences[current_id] = seq
    
    labels = pd.read_csv(label_file)
    label_dict = {row['pdb']: eval(row['locs']) for index, row in labels.iterrows()} 
    
    if use_cuda:
        esm_model = esm_model.to('cuda')

    os.makedirs(esm_cache_path, exist_ok=True)

    embeddings, label_vectors = {}, {}
    for pdb_id, seq in sequences.items():
        embedding_path = os.path.join(esm_cache_path, f"{pdb_id}.pt")
        if os.path.exists(embedding_path):
            embeddings[pdb_id] = torch.load(embedding_path)
        else:
            _, _, tokens = esm_alphabet.get_batch_converter()([(pdb_id, seq)])
            if use_cuda:
                tokens = tokens.to('cuda', non_blocking=True)
            
            with torch.no_grad():
                out = esm_model(tokens, repr_layers=[repr_layer], return_contacts=False)
            embeddings[pdb_id] = out['representations'][repr_layer].squeeze().cpu()
            torch.save(embeddings[pdb_id], embedding_path)
        
        label_vector = torch.zeros(len(seq), dtype=torch.float32)
        if pdb_id in label_dict:
            for site in label_dict[pdb_id]:
                if site-1 < len(seq):
                    label_vector[site-1] = 1.0
        label_vectors[pdb_id] = label_vector

    embeddings_list, label_vectors_list = [], []
    for pdb_id in embeddings.keys():
        embeddings_list.extend(embeddings[pdb_id])
        label_vectors_list.extend(label_vectors[pdb_id])
    return ProteinDataset(embeddings_list, label_vectors_list)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x).squeeze()

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Training Loss: {total_loss}')
        total_loss = 0
        with torch.no_grad():
            for embeddings, labels in test_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        print(f'Epoch {epoch+1}, Test Loss: {total_loss}')

def predict(model, sequence):
    embedding = torch.tensor(sequence)
    model.eval()
    with torch.no_grad():
        prediction = model(embedding)
    return prediction

if __name__ == "__main__":
    input_size = 480
    hidden_size = 512
    output_size = 1
    learning_rate = 0.001
    batch_size = 32
    epochs = 10
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet("esm2_t12_35M_UR50D")
    dataset = load_embeddings_and_labels('protein_sequences.fasta', 'data/csa_functional_sites.csv', esm_model, esm_alphabet)
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    model = MLP(input_size, hidden_size, output_size).to(device)

    pos_weight = torch.tensor([dataset.labels.count(0.0)/dataset.labels.count(1.0)])
    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_model(model, train_loader, test_loader, criterion, optimizer, epochs=epochs)
    
    model.eval()

    correct_positive = 0
    correct_negative = 0
    total_positive = 0
    total_negative = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            predicted = (output > 0.5).float()
            
            correct_positive += ((predicted == 1) & (target == 1)).sum().item()
            correct_negative += ((predicted == 0) & (target == 0)).sum().item()
            total_positive += (target == 1).sum().item()
            total_negative += (target == 0).sum().item()

    positive_accuracy = 100 * correct_positive / total_positive if total_positive > 0 else 0
    negative_accuracy = 100 * correct_negative / total_negative if total_negative > 0 else 0

    print(f'Positive accuracy: {positive_accuracy:.2f}% ({correct_positive}/{total_positive})')
    print(f'Negative accuracy: {negative_accuracy:.2f}% ({correct_negative}/{total_negative})')