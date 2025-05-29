import networkx as nx
import matplotlib.pyplot as plt

G = nx.erdos_renyi_graph(10, 0.1)
pos = nx.circular_layout(G) 
nx.draw(G, pos=pos)
plt.axis('off')
plt.show()

G = nx.erdos_renyi_graph(10, 0.5)
pos = nx.circular_layout(G) 
nx.draw(G, pos=pos)
plt.axis('off')
plt.show()

G = nx.erdos_renyi_graph(10, 0.9)
pos = nx.circular_layout(G) 
nx.draw(G, pos=pos)
plt.axis('off')
plt.show()

G = nx.gnm_random_graph(3, 2)
pos = nx.circular_layout(G) 
nx.draw(G, pos=pos)
plt.axis('off')
plt.show()

G = nx.watts_strogatz_graph(10, 4, 0.5)
pos = nx.circular_layout(G) 
nx.draw(G, pos=pos)
plt.axis('off')
plt.show()

G = nx.watts_strogatz_graph(10, 4, 0)
pos = nx.circular_layout(G) 
nx.draw(G, pos=pos)
plt.axis('off')
plt.show()

G = nx.watts_strogatz_graph(10, 4, 1)
pos = nx.circular_layout(G) 
nx.draw(G, pos=pos)
plt.axis('off')
plt.show()

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, split_labels=True, add_negative_train_samples=False),
])

dataset = Planetoid('.', name='Cora', transform=transform)

train_data, val_data, test_data = dataset[0]

from torch_geometric.nn import GCNConv, VGAE

class Encoder(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv1 = GCNConv(dim_in, 2 * dim_out)
        self.conv_mu = GCNConv(2 * dim_out, dim_out)
        self.conv_logstd = GCNConv(2 * dim_out, dim_out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

model = VGAE(Encoder(dataset.num_features, 16)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index) + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

for epoch in range(301):
    loss = train()
    val_auc, val_ap = test(val_data)
    if epoch % 50 == 0:
        print(f'Epoch: {epoch:>3} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}')

val_auc, val_ap = test(val_data)
print(f'\nTest AUC: {val_auc:.4f} | Test AP: {val_ap:.4f}')

z = model.encode(test_data.x, test_data.edge_index)
adj = torch.where((z @ z.T) > 0.9, 1, 0)
print(adj)

import pandas as pd
import numpy as np
from tensorflow import one_hot

import deepchem as dc
from deepchem.models.optimizers import ExponentialDecay
from deepchem.models import BasicMolGANModel as MolGAN
from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops

_, datasets, _ = dc.molnet.load_tox21()
df = pd.DataFrame(datasets[0].ids, columns=['smiles'])
print(df)

max_atom = 15
molecules = [x for x in df['smiles'].values if Chem.MolFromSmiles(x).GetNumAtoms() < max_atom]

featurizer = dc.feat.MolGanFeaturizer(max_atom_count=max_atom)

features = []
for x in molecules:
    mol = Chem.MolFromSmiles(x)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    feature = featurizer.featurize(mol)
    if feature.size != 0:
        features.append(feature[0])

# Remove invalid molecules
features = [x for x in features if type(x) is GraphMatrix]

# Create MolGAN
gan = MolGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000), vertices=max_atom)

# Create dataset
dataset = dc.data.NumpyDataset(X=[x.adjacency_matrix for x in features], y=[x.node_features for x in features])
print(dataset)

def iterbatches(epochs):
    for i in range(epochs):
        for batch in dataset.iterbatches(batch_size=gan.batch_size, pad_batches=True):
            adjacency_tensor = one_hot(batch[0], gan.edges)
            node_tensor = one_hot(batch[1], gan.nodes)
            yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]: node_tensor}

# Train model
gan.fit_gan(iterbatches(25), generator_steps=0.2, checkpoint_interval=5000)

# Generate 1000 samples
generated_data = gan.predict_gan_generator(1000)
generated_mols = featurizer.defeaturize(generated_data)

# Check molecule validity (unstable so you might end up with 0 valid molecules)
valid_mols = [x for x in generated_mols if x is not None]
print (f'{len(valid_mols)} valid molecules (out of {len((generated_mols))} generated molecules)')

generated_smiles = [Chem.MolToSmiles(x) for x in valid_mols]
generated_smiles_viz = [Chem.MolFromSmiles(x) for x in set(generated_smiles)]
print(f'{len(generated_smiles_viz)} unique valid molecules ({len(generated_smiles)-len(generated_smiles_viz)} redundant molecules)')

Draw.MolsToGridImage(generated_smiles_viz, molsPerRow=5, subImgSize=(200, 200), returnPNG=False)
plt.axis('off')
plt.show()