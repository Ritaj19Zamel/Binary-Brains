import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear
from pylab import rcParams
import pandas as pd
from torch_geometric.datasets import Twitch
from torch_geometric.nn import GNNExplainer
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
torch.manual_seed(42)
# Dataset source: https://github.com/benedekrozemberczki/datasets#twitch-social-networks
graph = Twitch(root=".", name="EN")[0]

rcParams['figure.figsize'] = 5, 5
df = pd.DataFrame(graph.y.numpy(), columns=["explicit_language"])
df['explicit_language'].value_counts().plot(kind='bar')
embedding_size = 128


class GNN(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GNN, self).__init__()

        # GCN layers
        self.initial_conv = GATConv(graph.num_features, embedding_size)
        self.conv1 = GATConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size, 1)

    def forward(self, x, edge_index):
        emb = F.relu(self.initial_conv(x, edge_index))
        emb = F.relu(self.conv1(emb, edge_index))
        return self.out(emb)


model = GNN()


def train_task_masks():
    num_nodes = graph.x.shape[0]
    ones = torch.ones(num_nodes)
    ones[4000:] = 0
    graph.train_mask = ones.bool()
    graph.test_mask = ~graph.train_mask.bool()

    print("Train nodes: ", sum(graph.train_mask))
    print("Test nodes: ", sum(graph.test_mask))
    graph.train_mask
    graph.test_mask


# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GNN()
model = model.to(device)
graph = graph.to(device)

# Loss function
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(graph.x, graph.edge_index)
    preds = out[graph.train_mask]
    targets = torch.unsqueeze(graph.y[graph.train_mask], 1)
    loss = loss_fn(preds.float(), targets.float())
    loss.backward()
    optimizer.step()
    return loss


def test():
    model.eval()
    optimizer.zero_grad()
    out = model(graph.x, graph.edge_index)
    preds = out[graph.test_mask]
    targets = torch.unsqueeze(graph.y[graph.test_mask], 1)
    loss = loss_fn(preds.float(), targets.float())
    return loss


for epoch in range(0, 800):
    tr_loss = train()
    if epoch % 100 == 0:
        loss = test()
        print(
            f'Epoch: {epoch:03d}, Test loss: {loss:.4f} | Train loss: {tr_loss:.4f}')


def predict():
    df = pd.DataFrame()
    # Model predictions'
    out = torch.sigmoid(model(graph.x, graph.edge_index))
    df["preds"] = out[graph.test_mask].round(
    ).int().cpu().detach().numpy().squeeze()
    df["prob"] = out[graph.test_mask].cpu().detach().numpy().squeeze().round(2)

    # Groundtruth
    df["gt"] = graph.y[graph.test_mask].cpu().detach().numpy().squeeze()

    print("Test ROC: ", roc_auc_score(df["gt"], df["preds"]))
    df.head(10)


def initialize_explainer():
    explainer = GNNExplainer(model, epochs=200, return_type='log_prob')

    # Explain node
    node_idx = 7
    node_feat_mask, edge_mask = explainer.explain_node(
        node_idx, graph.x, graph.edge_index)
    print("Size of explanation: ", sum(edge_mask > 0))
    print("Node features: ", graph.x[node_idx])
    print("Node label: ",  df["gt"][node_idx])
    print("Node prediction: ",  df["preds"][node_idx])


def show_mask(node_feat_mask, edge_mask):
    print(node_feat_mask.shape)
    print(edge_mask.shape)


def Visualize(explainer, node_idx, edge_mask):
    rcParams['figure.figsize'] = 10, 10
    # Visualize result
    ax, G = explainer.visualize_subgraph(
        node_idx, graph.edge_index, edge_mask, y=graph.y)
    plt.show()
