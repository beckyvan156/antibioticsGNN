

import pandas as pd
import io
import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import csv

from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from torchsummary import summary


""" Import data"""

drug = pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/drug.csv')
drug=drug.drop(drug.columns[0], axis=1)

edge=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/edge.csv')
edge=edge.drop(edge.columns[0], axis=1)

node=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/node.csv')

#phenotype=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/phenotype.csv')

phenotype=pd.read_csv('C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\Data\\purified_data\\antibiogram_phenotype.csv')



""" transform phenotype values to 0/1 """

# Replace 'resistant' with 1 and 'susceptible' with 0
phenotype = phenotype.replace({'resistant': 1, 'susceptible': 0})


""" Extract certain bacteria and antibiotics"""
# List of bacterial and antibiotics to include
include_bacteria = ['Klebsiella pneumoniae', 'Klebsiella aerogenes']
include_antibiotics = ['cefotaxime', 'piperacillin-tazobactam']

# Boolean masks for bacterial and antibiotics
mask_bacterial = phenotype['bacteria'].isin(include_bacteria)
mask_antibiotics = phenotype['antibiotics'].isin(include_antibiotics)

# Applying the masks to the dataframe
phenotype = phenotype[mask_bacterial & mask_antibiotics]


drug=drug[drug.index.isin(phenotype.index)]

node = node.rename(columns={node.columns[0]: 'sample_id'})
node=node[node['sample_id'].isin(phenotype['sample_id'])]


edge=edge[edge.index.isin(node.index)]
sample_ids=node['sample_id']
str_sample_ids = [str(i) for i in sample_ids]
edge=edge[str_sample_ids]



""" Get node names, node & drug combinations """

node_name=node[node.columns[0]]

node=node.drop(node.columns[0], axis=1)

node_drug_combination=drug[drug.columns[0]]

drug=drug.drop(drug.columns[0], axis=1)


" remove columns in node that are all 0"
node = node.loc[:, (node != 0).any(axis=0)]




"""**Transform Dataframe to Tensors**"""
phenotype=phenotype.iloc[:,-1]
# Convert the DataFrame to a numpy array
numpy_array = phenotype.values
# Convert the numpy array to a PyTorch Tensor
phenotype_tensor = torch.from_numpy(numpy_array)
print(phenotype_tensor)



# Convert the DataFrame to a numpy array
numpy_array = drug.values
# Convert the numpy array to a PyTorch Tensor
drug_tensor = torch.from_numpy(numpy_array)
print(drug_tensor)
drug_tensor =drug_tensor.float()



# Convert the DataFrame to a numpy array
numpy_array = edge.values
# Convert the numpy array to a PyTorch Tensor
edge_tensor = torch.from_numpy(numpy_array)
edge_tensor = edge_tensor.float()
print(edge_tensor)



# Convert the DataFrame to a numpy array
numpy_array = node.values
# Convert the numpy array to a PyTorch Tensor
node_tensor = torch.from_numpy(numpy_array)
print(node_tensor)
node_tensor = node_tensor.float()




"""**Transform distance matrix to edge index**"""

# Mask for the upper triangular part, excluding diagonal
mask = torch.triu(torch.ones_like(edge_tensor), diagonal=1)

# Extract the upper triangular elements
upper_triangular = edge_tensor * mask

# Get the indices of non-zero elements
indices = torch.nonzero(upper_triangular, as_tuple=True)

# Count the non-zero elements
count = torch.nonzero(upper_triangular).size(0)
print(count)

# Create the edge_index tensor
edge_index = torch.stack(indices, dim=0)

print(edge_index)
print(edge_index.shape)
print(edge_index.max().item())



""" **Sampling Graph Data to Training and Testing Datasets**"""

# Initialize the graph
# using the long() method
edge_long = edge_index.long()
data = Data(x=node_tensor, edge_index=edge_long)
print(data)


# Assuming 'data' is your full graph

# Randomly sample 266 (80%) nodes as training data
train_node_idx = torch.randperm(333)[:266]
print(train_node_idx.max().item())


# Extract subgraph
train_sub_node_features = data.x[train_node_idx]
train_sub_edge_index, _ = subgraph(train_node_idx, data.edge_index, relabel_nodes=False)

print(train_sub_node_features)
print(train_sub_edge_index)
print(train_node_idx)


# Generate the array
array = torch.arange(0, 333)
print(array)

# determine test data node idx
test_node_idx = array[np.isin(array, train_node_idx, invert=True)]
print(test_node_idx)

# use this boolean mask to select the desired elements from the data
test_sub_node_features = data.x[test_node_idx]
test_sub_edge_index, _ = subgraph(test_node_idx, data.edge_index, relabel_nodes=False)



node_name=torch.tensor(node_name.values)
train_node_name=node_name[train_node_idx]
print(train_node_name)
test_node_name=node_name[test_node_idx]
print(test_node_name)



""" ** Subtract training and testing datasets for Drug, Phenotype data ** """

# Convert to lists for easy comparison
node_drug_combination_list = node_drug_combination.tolist()
train_node_name_list = train_node_name.tolist()

# Use list comprehension to find indices in A that match names in B
train_drug_indices = [idx for idx, name in enumerate(node_drug_combination_list) if name in train_node_name_list]
train_drug_indices=torch.tensor(train_drug_indices)
print(train_drug_indices)


# Convert to lists for easy comparison
test_node_name_list = test_node_name.tolist()

# Use list comprehension to find indices in A that match names in B
test_drug_indices = [idx for idx, name in enumerate(node_drug_combination_list) if name in test_node_name_list]
test_drug_indices=torch.tensor(test_drug_indices)
print(test_drug_indices)

train_drug_tensor=drug_tensor[train_drug_indices]
test_drug_tensor=drug_tensor[test_drug_indices]
print(train_drug_tensor)
print(test_drug_tensor)

train_phenotype_tensor=phenotype_tensor[train_drug_indices]
test_phenotype_tensor=phenotype_tensor[test_drug_indices]
print(train_phenotype_tensor)




train_data = Data(x=train_sub_node_features, edge_index=train_sub_edge_index,
                  y=train_phenotype_tensor,condition=train_drug_tensor,
                  index=train_node_idx,node_name=train_node_name_list)

print(train_data)
print(train_data.edge_index)



test_data = Data(x=test_sub_node_features, edge_index=test_sub_edge_index,
                 y=test_phenotype_tensor,condition=test_drug_tensor,
                 index=test_node_idx,node_name=test_node_name_list)

print(test_data)





""" ** Wrap training data (from Graph) to DataLoader ** """

def get_y_indices_for_x(x_index_range):
    node_name_i=node_name[x_index_range]
    node_name_i=node_name_i.tolist()
    drug_indices = [idx for idx, name in enumerate(node_drug_combination_list) if name in node_name_i]
    drug_indices=torch.tensor(drug_indices)  
    return drug_indices

# add edge to isolated nodes connecting to themselves, allowing you to utilize the subgraph() function without errors.#
def add_self_loops(edge_index, num_nodes):
    # Generate self-loops
    self_loops = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)
    
    # Concatenate edge_index with self-loops
    edge_index_with_loops = torch.cat([edge_index, self_loops], dim=1)
    
    return edge_index_with_loops

# Assuming you have edge_index and num_nodes (total number of nodes in the graph)
edge_index_with_loops = add_self_loops(edge_index, 333)
print(edge_index_with_loops)
print(edge_index_with_loops.shape)




def split_graph_into_subgraphs(data, num_subgraphs=10):
    subgraphs = []
    
    # Randomly permute the indices
    total_nodes = data.x.shape[0]
    permuted_indices = torch.randperm(total_nodes)
    
    nodes_per_subgraph = total_nodes // num_subgraphs
    
    for i in range(num_subgraphs):
        start_idx = i * nodes_per_subgraph
        end_idx = (i + 1) * nodes_per_subgraph
        
        # Use the permuted indices to get the actual indices for x and y
        current_indices = permuted_indices[start_idx:end_idx]
        
        node_index=data.index[current_indices]
        
        sub_node=node_tensor[node_index]
        sub_edge_index, _ = subgraph(node_index,edge_index_with_loops, relabel_nodes=True)

        drug_indices=get_y_indices_for_x(node_index)
        drug_tensor_i=drug_tensor[drug_indices]
        phenotype_i=phenotype_tensor[drug_indices]
        
        
        sub_data = Data(x=sub_node, edge_index=sub_edge_index,
                          y=phenotype_i,condition=drug_tensor_i)
        subgraphs.append(sub_data)
    
    return subgraphs




# Create subgraphs
subgraphs= split_graph_into_subgraphs(train_data, num_subgraphs=16)

# Now use the DataLoader
from torch_geometric.data import DataLoader
train_loader = DataLoader(subgraphs, batch_size=4, shuffle=True)


batch = next(iter(train_loader))
single_data_instance = batch[0]
print(single_data_instance)
print(single_data_instance.edge_index)


"relabel edge index for complete training and testing data"
train_data.edge_index, _ = subgraph(train_node_idx, data.edge_index, relabel_nodes=True)               
print(train_data.edge_index)


test_data.edge_index, _ = subgraph(test_node_idx, data.edge_index, relabel_nodes=True)               
print(test_data.edge_index)



""" **************************    Write FiLM   ***********************************  """


""" 1). Build Linear Layers to Get Drug Embeddings for training and test data"""


""" 2). Build FiLM layers """

class FiLMedGNN(torch.nn.Module):
    def __init__(self, hidden_dim, dropout=0.3, node_dim=2090,condition_dim=29):
        super(FiLMedGNN, self).__init__()
        self.hidden_dim=hidden_dim
        self.condition_dim=condition_dim
        self.node_dim=node_dim
        
        
        self.conv1 = GCNConv(self.node_dim, self.hidden_dim)
  
        # Global FiLM Parameters
        self.linear_gamma = Linear(self.condition_dim, self.hidden_dim)
        self.linear_beta = Linear(self.condition_dim, self.hidden_dim)

        # This will be used for condition-level predictions
        self.out_proj = Linear(hidden_dim, 2)

        # add dropout to prevent overfitting
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index, condition = data.x, data.edge_index,  data.condition
        
        #condition=linear_model(condition)

        # Generate FiLM parameters
        gamma = self.linear_gamma(condition)
        beta = self.linear_beta(condition)

        # Apply GNN layers with FiLM
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        
        x=x.transpose(0, 1)
        x = torch.matmul(gamma,x)
        projection = Linear(x.shape[1], self.hidden_dim)
        x=projection(x) 
        x=x+beta


        x = torch.relu(x)
        x = self.dropout(x)
        # x = self.conv2(x, edge_index)
        # x = torch.relu(x)

        # final prediction
        out = self.out_proj(x)

        # apply softmax to the output
        out =F.log_softmax(out, dim=1)



        return out





vals = { 'hidden_dim':29, 
         'dropout':0.4, 'node_dim':2090, 'condition_dim':29}

model = FiLMedGNN(**vals)

#single_data_instance = batch[0]
#print(single_data_instance)
#output = model(single_data_instance)
#output
#F.nll_loss(output, single_data_instance.y)
#single_data_instance.y



"""  *****************************   Train the model   ****************************** """ 

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


""" 1). Define traing and test functions """

def train():
    
    model.train()
    total_loss = 0.0

    # Iterate over batches
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        # Accumulate the total loss
        total_loss += loss.item()
        optimizer.step()
    return total_loss


def test(input_data):

    model.eval()
    output = model(input_data)
    pred = output.max(dim=1)[1]
    correct = pred.eq(input_data.y).sum().item()
    return correct / len(input_data.y)


# Function to save test data predictions
folder_path='C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\repo'
def save_predictions(input_data, epoch):
    model.eval()

    # Store probabilities here
    all_probs = []
    
    # Make sure to disable gradients for evaluation
    with torch.no_grad():
        output = model(input_data)
        
        # Apply exponential to log-softmax output to get probabilities
        probabilities = torch.exp(output)
        
        # Add probabilities to list (detach tensor and convert them to list)
        all_probs.extend(probabilities.cpu().numpy().tolist())
        
    # Convert the list to a DataFrame
    df = pd.DataFrame(all_probs, columns=[f'Class_{i}' for i in range(output.size(1))])
    
    # Save the DataFrame to a CSV file
    df.to_csv(f'{folder_path}\\epoch_{epoch}_probabilities.csv', index=False)



""" 3). Train the model  """

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# create lists to save train ACC and test ACC for each epoch
train_accuracies = []
test_accuracies = []



for epoch in range(101):
    loss = train()

    train_acc = test(input_data=train_data)
    train_accuracies.append(train_acc)

    test_acc = test(input_data=test_data)
    test_accuracies.append(test_acc)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, 'f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    # Save test data predictions at epoch 50 and 80
    if epoch == 10 or epoch == 100:
        save_predictions(input_data=test_data, epoch=epoch)
        
        # Print gradients
        print(f"\nGradients at epoch {epoch}:")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.norm().item()}")
        print("\n")
        
    
""" Visualize train and test ACC """
plt.figure(figsize=(12, 6))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy across Epochs: rate=0.05')
plt.legend()
plt.grid(True)
plt.show() 



