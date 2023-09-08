# -*- coding: utf-8 -*-
"""antibiotics project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TCEKjdorb8d-RejJfteTWdUBZa8U_zxs
"""
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




""" Import data"""

drug = pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/drug.csv')
drug=drug.drop(drug.columns[0], axis=1)

edge=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/edge.csv')
edge=edge.drop(edge.columns[0], axis=1)

node=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/node.csv')

phenotype=pd.read_csv('C:/Users/becky/OneDrive/Desktop/2023 summer intern/Data/purified_data/phenotype.csv')




""" transform phenotype values to 0/1 """

# Replace 'resistant' with 1 and 'susceptible' with 0
phenotype = phenotype.replace({'resistant': 1, 'susceptible': 0})




""" Get node names, node & drug combinations """

node_name=node[node.columns[0]]

node=node.drop(node.columns[0], axis=1)

node_drug_combination=drug[drug.columns[0]]

drug=drug.drop(drug.columns[0], axis=1)



"""**Transform Dataframe to Tensors**"""

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

# Randomly sample 2714 (80%) nodes as training data
train_node_idx = torch.randperm(3393)[:2714]
print(train_node_idx.max().item())

# Extract subgraph
train_sub_node_features = data.x[train_node_idx]
train_sub_edge_index, _ = subgraph(train_node_idx, data.edge_index, relabel_nodes=False)

print(train_sub_node_features)
print(train_sub_edge_index)
print(train_node_idx)


# Generate the array
array = torch.arange(0, 3393)
print(array)

# determine test data node idx
test_node_idx = array[np.isin(array, train_node_idx, invert=True)]
print(test_node_idx)

# use this boolean mask to select the desired elements from the data
test_sub_node_features = data.x[test_node_idx]
test_sub_edge_index, _ = subgraph(test_node_idx, data.edge_index, relabel_nodes=False)



node_name=torch.tensor(node_name)
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
edge_index_with_loops = add_self_loops(edge_index, 3393)
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
subgraphs= split_graph_into_subgraphs(train_data, num_subgraphs=32)

# Now use the DataLoader
from torch_geometric.data import DataLoader
train_loader = DataLoader(subgraphs, batch_size=4, shuffle=True)


batch = next(iter(train_loader))
single_data_instance = batch[0]
print(single_data_instance)
print(single_data_instance.edge_index)










""" **************************    Write FiLM   ***********************************  """


""" 1). Build Linear Layers to Get Drug Embeddings for training and test data"""

#%% Linear layer
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=True)

    def forward(self, x):
        out = self.linear(x)
        return out


linear_model = LinearModel(29,256) ## to get same column dimension as graph embedding




""" 2). Build FiLM layers """

class FiLMedGNN(torch.nn.Module):
    def __init__(self, hidden_dim, dropout=0.5):
        super(FiLMedGNN, self).__init__()
        self.conv1 = GCNConv(14615, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Global FiLM Parameters
        self.linear_gamma = Linear(256, hidden_dim)
        self.linear_beta = Linear(256, hidden_dim)

        # This will be used for condition-level predictions
        self.out_proj = Linear(hidden_dim, 2)

        # add dropout to prevent overfitting
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data,hidden_dim):
        x, edge_index, condition = data.x, data.edge_index,  data.condition
        
        condition=linear_model(condition)

        # Generate FiLM parameters
        gamma = self.linear_gamma(condition)
        beta = self.linear_beta(condition)

        # Apply GNN layers with FiLM
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        
        x=x.transpose(0, 1)
        x = torch.matmul(gamma,x)
        projection = Linear(x.shape[1], hidden_dim)
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




model = FiLMedGNN(hidden_dim=128)



# debug #

# single_data_instance = batch[0]
# print(single_data_instance)

# x, edge_index, condition = train_data.x, train_data.edge_index,  train_data.condition

# print(edge_index)


# conv1 = GCNConv(14615, 128)
# conv2 = GCNConv(128, 128)
# linear_gamma = Linear(128, 128)
# linear_beta = Linear(128, 128)
# out_proj = Linear(128, 2)
# condition=linear_model.forward(condition)
# gamma =linear_gamma(condition)
# beta = linear_beta(condition)
# x = conv1(x, edge_index)
# x=x.transpose(0, 1)
# x = torch.matmul(gamma,x)
# projection = nn.Linear(x.shape[1], 128)
# x=projection(x)
# x=x+beta


# x = torch.relu(x)

# dropout = nn.Dropout(p=0.1)
 
# x = dropout(x)
# x = conv2(x, edge_index)
# x = torch.relu(x)

# # Concatenate condition features and graph embedding, and pass it through the output layer
# out = out_proj(x)
# out =F.log_softmax(out, dim=1)
# print(out)
# loss = F.nll_loss(out, single_data_instance.y.squeeze(1))



"""  *****************************   Train the model   ****************************** """ 



""" 1). Define traing and test functions """

def train():
    
    model.train()
    total_loss = 0.0

    # Iterate over batches
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data,hidden_dim=128)
        loss = F.nll_loss(output, data.y.squeeze(1))
        loss.backward()
        # Accumulate the total loss
        total_loss += loss.item()
        optimizer.step()
    return total_loss






def test(input_data):

    model.eval()
    output = model(input_data,hidden_dim=128)
    pred = output.max(dim=1)[1]
    correct = pred.eq(input_data.y.squeeze(1)).sum().item()
    return correct / len(input_data.y)



# node_features = train_data.x.shape[1]
# hidden_dim = 128
# embedding_dim = drug_embedding.shape[1]
# num_conditions = drug_embedding.shape[0]
# num_nodes = 32
# num_edges = train_data.edge_index.shape[1]
# num_classes = 2

# condition = drug_embedding


# # Initialize the model
# model = FiLMedGNN(node_features, hidden_dim, embedding_dim, num_nodes, num_classes)

# loss = train(node_features=train_data.x.shape[1], hidden_dim=128, embedding_dim=drug_embedding.shape[1],
#                  num_conditions=drug_embedding.shape[0], num_nodes=32, num_edges=train_data.edge_index.shape[1],
#                  num_classes=2,condition=drug_embedding)

""" 3). Train the model  """

"relabel edge index for complete training and testing data"
train_data.edge_index, _ = subgraph(train_node_idx, data.edge_index, relabel_nodes=True)               
print(train_data.edge_index)


test_data.edge_index, _ = subgraph(test_node_idx, data.edge_index, relabel_nodes=True)               
print(test_data.edge_index)




# Move the model to the appropriate device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)




# Function to save test data predictions
folder_path='C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern'
def save_predictions(input_data, epoch):
    model.eval()

    # Store probabilities here
    all_probs = []
    
    # Make sure to disable gradients for evaluation
    with torch.no_grad():
        output = model(input_data, hidden_dim=128)
        
        # Apply exponential to log-softmax output to get probabilities
        probabilities = torch.exp(output)
        
        # Add probabilities to list (detach tensor and convert them to list)
        all_probs.extend(probabilities.cpu().numpy().tolist())
        
    # Convert the list to a DataFrame
    df = pd.DataFrame(all_probs, columns=[f'Class_{i}' for i in range(output.size(1))])
    
    # Save the DataFrame to a CSV file
    df.to_csv(f'{folder_path}\\epoch_{epoch}_probabilities.csv', index=False)


    
    
    
    

for epoch in range(101):
    loss = train()
    
    # After training for the epoch, print the weights
    # print(f"Weights after epoch {epoch+1}:")
    
    # state_dict = model.state_dict()
    # for key, value in state_dict.items():
    #     print(key, value.shape)
    #     print(value)
    #     print("----------------------------")

    train_acc = test(input_data=train_data)

    test_acc = test(input_data=test_data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    # Save test data predictions at epoch 50 and 80
    if epoch == 50 or epoch == 80:
        save_predictions(input_data=test_data, epoch=epoch)



""" Calculate Test ACC by species and antibiotics """
test_epoch80=pd.read_csv('C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\epoch_80_probabilities.csv')
phenotype=pd.read_csv('C:\\Users\\becky\\OneDrive\\Desktop\\2023 summer intern\\Data\\purified_data\\antibiogram_phenotype.csv')

index=test_drug_indices.tolist()
test_phenotype=phenotype.iloc[index]

result = pd.concat([test_phenotype.reset_index(drop=True), test_epoch80.reset_index(drop=True)], axis=1)


# Add classifier column
result['prediction'] = np.where(result['Class_0'] > 0.5, 'susceptible', 'resistant')
result['correct']=result['phenotype']==result['prediction']

 
# Group by 'bacteria' and 'antibiotics' and calculate accuracy
accuracy = result.groupby(['bacteria', 'antibiotics'])['correct'].mean().reset_index()

# Convert accuracy to percentage
accuracy['correct'] = accuracy['correct'] * 100

accuracy.to_csv(f'{folder_path}\\test_accuracy.csv', index=False)





"""   Simplified Models: node feature + drug feature with a linear layer   """

" 1). feature padding of drug feature to match column dimension of node feature"
"     feature padding of node feature to match row dimension of drug feature"
# add_dimension_column=train_sub_node_features.shape[1]-train_drug_tensor.shape[1]
# train_drug_tensor_padded = F.pad(train_drug_tensor, (0,add_dimension_column)) 
# test_drug_tensor_padded = F.pad(test_drug_tensor, (0, add_dimension_column)) 

add_dimension_row_train=train_drug_tensor.shape[0]-train_sub_node_features.shape[0]
train_sub_node_features_padded = F.pad(train_sub_node_features, (0,0,0,add_dimension_row_train)) 

add_dimension_row_test=test_drug_tensor.shape[0]-test_sub_node_features.shape[0]
test_sub_node_features_padded = F.pad(test_sub_node_features, (0,0,0, add_dimension_row_test)) 

print(train_sub_node_features_padded.shape)
# print(train_drug_tensor_padded.shape)

print(test_sub_node_features_padded.shape)
# print(test_drug_tensor_padded.shape)


" 2). concetenate 2 tensors by columns"
simple_train= torch.cat((train_sub_node_features_padded, train_drug_tensor), dim=1)
simple_test=torch.cat((test_sub_node_features_padded, test_drug_tensor), dim=1)



" 3). import train data to Dataloader"
from torch_geometric.data import DataLoader

train_combined=Data(x=simple_train,y=train_phenotype_tensor)
train_combined_list = [train_combined]  # Wrap the Data object in a list
train_loader = DataLoader(train_combined_list, batch_size=32, shuffle=True)

def split_graph_into_subgraphs(x, y, num_subgraphs=10):
    subgraphs = []
    
    # Randomly permute the indices
    total_nodes = x.size(0)
    permuted_indices = torch.randperm(total_nodes)
    
    nodes_per_subgraph = total_nodes // num_subgraphs
    
    for i in range(num_subgraphs):
        start_idx = i * nodes_per_subgraph
        end_idx = (i + 1) * nodes_per_subgraph
        
        # Use the permuted indices to get the actual indices for x and y
        current_indices = permuted_indices[start_idx:end_idx]
        
        sub_x = x[current_indices]
        sub_y = y[current_indices]
        
        sub_data = Data(x=sub_x, y=sub_y)
        subgraphs.append(sub_data)
        
    return subgraphs

# Create subgraphs
subgraphs = split_graph_into_subgraphs(simple_train, train_phenotype_tensor, num_subgraphs=64)

# Now use the DataLoader
train_loader = DataLoader(subgraphs, batch_size=16, shuffle=True)

batch = next(iter(train_loader))
single_data_instance = batch[0]
print(single_data_instance.x)


" 4). apply a linear layer"
class linearmodel(nn.Module):
    def __init__(self, input_dim, output_dim=1):  # Default to 1 for binary outcome
        super(linearmodel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=True)

    def forward(self, x):
        out = self.linear(x)
        return out





" 5). define train and test functions"
model = linearmodel(14644)

# BCEWithLogitsLoss combines sigmoid with BCELoss
# If you're using this, you should remove the sigmoid activation from the model
criterion = torch.nn.BCEWithLogitsLoss() 

def train():
    model.train()

    loss_all = 0
    # # Create subgraphs
    # subgraphs = split_graph_into_subgraphs(simple_train, train_phenotype_tensor, num_subgraphs=32)

    # # Now use the DataLoader
    # train_loader = DataLoader(subgraphs, batch_size=32, shuffle=True)

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x)
        # Ensure that the targets are float type, expected by BCEWithLogitsLoss
        loss = criterion(output.squeeze(1), data.y.squeeze().float())       
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all 

def test_trainacc():

    model.eval()
    output = model(train_combined.x)
    pred = (torch.sigmoid(output) > 0.5).long()  # Convert to 0/1 predictions
    correct = pred.eq(train_combined.y.long()).sum().item()
    return correct / len(train_phenotype_tensor)


def test():
    model.eval()
    output = model(simple_test)
    pred = (torch.sigmoid(output) > 0.5).long()  # Convert to 0/1 predictions
    correct = pred.eq(test_phenotype_tensor.long()).sum().item()
    return correct / len(test_phenotype_tensor)



" 6). traing the model for 400 epoch"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(1, 401):
    loss = train()
    
    train_acc = test_trainacc()
    
    test_acc = test()
    
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')





















""" Debug """

" 3). import train data to Dataloader"
from torch_geometric.data import DataLoader

train_combined=Data(x=simple_train,y=train_phenotype_tensor)
train_combined_list = [train_combined]  # Wrap the Data object in a list
train_loader = DataLoader(train_combined_list, batch_size=32, shuffle=True)

def split_graph_into_subgraphs(x, y, num_subgraphs=10):
    subgraphs = []
    nodes_per_subgraph = x.size(0) // num_subgraphs
    
    for i in range(num_subgraphs):
        start_idx = i * nodes_per_subgraph
        end_idx = (i + 1) * nodes_per_subgraph
        
        sub_x = x[start_idx:end_idx]
        sub_y = y[start_idx:end_idx]
        
        sub_data = Data(x=sub_x, y=sub_y)
        subgraphs.append(sub_data)
    
    return subgraphs

# Create subgraphs
subgraphs = split_graph_into_subgraphs(simple_train, train_phenotype_tensor, num_subgraphs=64)

# Now use the DataLoader
train_loader = DataLoader(subgraphs, batch_size=16, shuffle=True)

batch = next(iter(train_loader))
single_data_instance = batch[0]
print(single_data_instance.y.squeeze(1))







next(iter(train_loader))

batch_indices,batch_x, batch_edge_index = next(iter(loader))
print(batch_x.shape)
print(batch_edge_index.shape)
# Extract subgraph
batch_edge_index, _ = subgraph(batch_indices, train_data.edge_index, relabel_nodes=True)
print(batch_edge_index)
print(batch_edge_index.shape)


batch_data = Data(x=batch_x, edge_index=batch_edge_index)

# Now let's create some dummy data and conditions for demonstration
node_features = train_data.x.shape[1]
hidden_dim = 128
embedding_dim = drug_embedding.shape[1]
num_conditions = drug_embedding.shape[0]
num_nodes =32
num_edges = train_data.edge_index.shape[1]
num_classes = 2

# Generate FiLM parameters
condition = drug_embedding
linear_gamma = Linear(embedding_dim, num_nodes)
linear_beta = Linear(embedding_dim, hidden_dim)
gamma = linear_gamma(condition)
beta =linear_beta(condition)

print(gamma.shape)
print(beta.shape)

"""**Extreact edge index from batch_x**"""

# Apply GNN layers with FiLM
conv1 = GCNConv(14615, 128)
conv2 = GCNConv(hidden_dim, hidden_dim)
x = conv1(batch_data.x, batch_data.edge_index)
x = torch.matmul(gamma,x) + beta  # FiLM modulation
x = torch.relu(x)

x = conv2(x, batch_data.edge_index)
x = torch.relu(x)

print(x.shape)

# # Global mean pooling
# graph_embedding = global_mean_pool(data.x, data.batch)
# graph_embedding = graph_embedding.view(-1)

# Concatenate condition features and graph embedding, and pass it through the output layer
# This will be used for condition-level predictions
out_proj = Linear(embedding_dim + hidden_dim, num_classes)
out = out_proj(torch.cat([condition, graph_embedding], dim=-1))

test=torch.cat([condition, x], dim=-1)
print(test.shape)

out_proj = Linear(embedding_dim + hidden_dim, num_classes)
out = out_proj(test)

print(out.shape)
print(out)



out = F.softmax(out, dim=1)  # apply softmax to the output
print(out)

