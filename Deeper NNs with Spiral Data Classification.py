"""
This program is meant to compare the performances of Neural Networks with various numbers of hidden layers and neurons
in PyTorch using a created spiral dataset. We can control the layers of the NN using lists. We iterate through the list
to determine different attributes of the neural network. The first item in the 'layer' list is the number of input
features. The last item in the 'layer' list is the number of output classes. Any item in between the denotes the number
of neurons in a hidden layer. The number of items between the first and last layer denotes the number of hidden layers.
The layers are controlled in the __main__ function. We plot the different combinations of hidden layers and neurons to
visualize the performance of the different neural network iterations.
"""

# Import Necessary Libraries.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader


# Create Data Class
class Data(Dataset):
    """This class creates the spiral data for the model to classify.
    Modified from: http://cs231n.github.io/neural-networks-case-study/
    """

    # Constructor
    def __init__(self, K=3, N=500):
        D = 2
        X = np.zeros((N * K, D))  # data matrix (each row = single example)
        y = np.zeros(N * K, dtype='uint8')  # class labels
        for j in range(K):
            ix = range(N * j, N * (j + 1))
            r = np.linspace(0.0, 1, N)  # radius
            t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.x = torch.from_numpy(X).type(torch.FloatTensor)
        self.len = y.shape[0]

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len

    # Plot the diagram
    def plot_stuff(self):
        plt.plot(self.x[self.y[:] == 0, 0].numpy(), self.x[self.y[:] == 0, 1].numpy(), 'o', label="y = 0")
        plt.plot(self.x[self.y[:] == 1, 0].numpy(), self.x[self.y[:] == 1, 1].numpy(), 'ro', label="y = 1")
        plt.plot(self.x[self.y[:] == 2, 0].numpy(), self.x[self.y[:] == 2, 1].numpy(), 'go', label="y = 2")
        plt.legend()
        plt.show()


# Create Net model class
class Net(nn.Module):
    """
    This class creates a neural network with an iterative method. The 'layers' input determines the shape of the NN.
    The 'layers' input is controlled in the __main__ method. The ReLU activation function is used.
    """

    # Constructor
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):  # Iterate through 'layer' list to determine NN layers
            if l < L - 1:
                activation = F.relu(linear_transform(activation))
            else:
                activation = linear_transform(activation)  # ReLU is not used on the output layer.
        return activation


# Define the function to plot the diagram
def plot_decision_regions_3class(model, data_set, title_):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    # cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    _, yhat = torch.max(model(XX), 1)
    yhat = yhat.numpy().reshape(xx.shape)
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light, shading='auto')
    plt.plot(X[y[:] == 0, 0], X[y[:] == 0, 1], 'ro', label='y=0')
    plt.plot(X[y[:] == 1, 0], X[y[:] == 1, 1], 'go', label='y=1')
    plt.plot(X[y[:] == 2, 0], X[y[:] == 2, 1], 'o', label='y=2')
    plt.title(title_)
    plt.legend()
    plt.show()


# Define the function for training the model
def train(data_set, model, criterion, train_loader, optimizer, epochs=100):
    LOSS = []
    ACC = []
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LOSS.append(loss.item())
        ACC.append(accuracy(model, data_set))

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(LOSS, color=color)
    ax1.set_xlabel('Iteration', color=color)
    ax1.set_ylabel('total loss', color=color)
    ax1.tick_params(axis='y', color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(ACC, color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()
    return LOSS


# The function to calculate the accuracy
def accuracy(model, data_set):
    _, yhat = torch.max(model(data_set.x), 1)
    return (yhat == data_set.y).numpy().mean()


# Run the program.
if __name__ == '__main__':
    """
    In this main method, we create an instance of the dataset and train various iterations of a neural network on it.
    The results are plotted. We can see that changing the number of hidden layers and neurons in each layer changes
    the performance of the model and decision boundaries. Too many or too few neurons or layers may cause over or 
    under fitting. 
    """

    # Create dataset object.
    data_set = Data()
    data_set.plot_stuff()
    data_set.y = data_set.y.view(-1)

    # Train the model with one Hidden Layer and 50 Neurons.
    layers_1 = [2, 50, 3]
    # The first item in the 'Layers' list is the number of input features and the last is the number of output
    # classes. Any number in between is the number of neurons in a hidden layer, so a list with three numbers has only
    # one hidden layer. Add more items in the list to create deeper neural networks. This is a simple way to create
    # neural networks without explicitly having to add layers to the network class. (Easier for deep NNs)

    model_1 = Net(layers_1)
    plot_decision_regions_3class(model_1, data_set, title_='One Hidden Layer with 50 Neurons')
    learning_rate = 0.10
    optimizer = torch.optim.SGD(model_1.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset=data_set, batch_size=20)
    criterion = nn.CrossEntropyLoss()
    LOSS = train(data_set, model_1, criterion, train_loader, optimizer, epochs=100)

    # Train the model with 2 hidden layers with 10 neurons each.
    layers_2 = [2, 10, 10, 3]  # For example there are two hidden layers, each with 10 neurons, an input with two
    # features and an output with three classes.

    model_2 = Net(layers_2)
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model_2.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset=data_set, batch_size=20)
    criterion = nn.CrossEntropyLoss()
    LOSS = train(data_set, model_2, criterion, train_loader, optimizer, epochs=1000)

    plot_decision_regions_3class(model_2, data_set, title_='Two Hidden Layers with 10 Neurons Each')

    # Train the model with 3 hidden layers with 10 neurons each.
    layers_3 = [2, 10, 10, 10, 3]
    model_3 = Net(layers_3)
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model_3.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset=data_set, batch_size=20)
    criterion = nn.CrossEntropyLoss()
    LOSS = train(data_set, model_3, criterion, train_loader, optimizer, epochs=1000)

    plot_decision_regions_3class(model_3, data_set, title_='Three Hidden Layers with 10 Neurons Each')