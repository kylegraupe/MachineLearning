"""
This program is meant to compare the different activation function performances for Neural Networks with two hidden
layers in PyTorch using the MNIST dataset. The inputs for the number of neurons in each hidden layer are controlled
in the __main__ method. There are ten output classes in each NN to represent each digit from 0 to 9. Each activation
function's training loss and validation loss are plotted against each other in order to visualize the comparison.
"""

# Import required libraries.
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np

# Create the model class using sigmoid as the activation function.
class Net(nn.Module):
    """This class creates a neural network with two hidden layers, with the number of neurons in each hidden layer
    and the number of output classes specified in the __main__ method. The activation function is a sigmoid function."""

    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)  # Hidden Layer 1
        self.linear2 = nn.Linear(H1, H2)  # Hidden Layer 2
        self.linear3 = nn.Linear(H2, D_out)  # Output Layer

    # Prediction
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))  # Sigmoid activation function
        x = torch.sigmoid(self.linear2(x))  # Sigmoid activation function
        x = self.linear3(x)
        return x


# Create the model class using Tanh as a activation function.
class NetTanh(nn.Module):
    """This class creates a neural network with two hidden layers, with the number of neurons in each hidden layer
    and the number of output classes specified in the __main__ method. The activation function is a 'tanh' function."""

    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        super(NetTanh, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)  # Hidden Layer 1
        self.linear2 = nn.Linear(H1, H2)  # Hidden Layer 2
        self.linear3 = nn.Linear(H2, D_out)  # Output Layer

    # Prediction
    def forward(self, x):
        x = torch.tanh(self.linear1(x))  # Tanh activation function
        x = torch.tanh(self.linear2(x))  # Tanh activation function
        x = self.linear3(x)
        return x


# Create the model class using Relu as a activation function.
class NetRelu(nn.Module):
    """This class creates a neural network with two hidden layers, with the number of neurons in each hidden layer
    and the number of output classes specified in the __main__ method. The activation function is a 'ReLU' function."""

    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        super(NetRelu, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)  # Hidden Layer 1
        self.linear2 = nn.Linear(H1, H2)  # Hidden Layer 2
        self.linear3 = nn.Linear(H2, D_out)  # Output Layer

    # Prediction
    def forward(self, x):
        x = torch.relu(self.linear1(x))  # ReLU activation function
        x = torch.relu(self.linear2(x))  # ReLU activation function
        x = self.linear3(x)
        return x


# Create function for model training.
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    i = 0
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.data.item())

        correct = 0
        for x, y in validation_loader:
            z = model(x.view(-1, 28 * 28))
            _, label = torch.max(z, 1)
            correct += (label == y).sum().item()

        accuracy = 100 * (correct / len(validation_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)

    return useful_stuff


# Create the training dataset from MNIST dataset.
def train_dset():
    train_dataset_ = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    return train_dataset_


# Create the validating dataset from MNIST dataset.
def val_dset():
    validation_dataset_ = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    return validation_dataset_


# Create the criterion function using Cross Entropy Loss.
def criterion_function():
    criterion = nn.CrossEntropyLoss()
    return criterion


# Create the training data loader and validation data loader objects.
def loaders():
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)
    return train_loader, validation_loader


# Train the model with sigmoid activation function.
def train_sigmoid(learning_rate, input_dim, hidden_dim1, hidden_dim2, output_dim, criterion, train_loader,
                  validation_loader, cust_epochs):
    model = Net(input_dim, hidden_dim1, hidden_dim2, output_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)
    return training_results


# Train the model with tanh activation function.
def train_tanh(learning_rate, input_dim, hidden_dim1, hidden_dim2, output_dim, criterion, train_loader,
               validation_loader, cust_epochs):
    model_tanh = NetTanh(input_dim, hidden_dim1, hidden_dim2, output_dim)
    optimizer = torch.optim.SGD(model_tanh.parameters(), lr=learning_rate)
    training_results_tanh = train(model_tanh, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)
    return training_results_tanh


# Train the model with relu activation function.
def train_relu(learning_rate, input_dim, hidden_dim1, hidden_dim2, output_dim, criterion, train_loader,
               validation_loader, cust_epochs):
    modelRelu = NetRelu(input_dim, hidden_dim1, hidden_dim2, output_dim)
    optimizer = torch.optim.SGD(modelRelu.parameters(), lr=learning_rate)
    training_results_relu = train(modelRelu, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)
    return training_results_relu


# Plot the training loss for all activation functions.
def plot_training_loss(training_results_tanh, training_results, training_results_relu):
    # Compare the training loss
    plt.plot(training_results_tanh['training_loss'], label='tanh')
    plt.plot(training_results['training_loss'], label='sigmoid')
    plt.plot(training_results_relu['training_loss'], label='relu')
    plt.ylabel('loss')
    plt.title('training loss iterations')
    plt.legend()
    plt.show()


# Plot the validation loss for all activation functions.
def plot_val_loss(training_results_tanh, training_results, training_results_relu):
    # Compare the validation loss
    plt.plot(training_results_tanh['validation_accuracy'], label='tanh')
    plt.plot(training_results['validation_accuracy'], label='sigmoid')
    plt.plot(training_results_relu['validation_accuracy'], label='relu')
    plt.ylabel('validation accuracy')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()


# Run the program.
if __name__ == '__main__':

    # Set the parameters for creating the model.
    input_dim_ = 28 * 28
    hidden_dim1_ = 50  # Number of Neurons in Hidden Layer
    hidden_dim2_ = 50  # Number of Neurons in Hidden Layer.
    output_dim_ = 10  # Number of output classes.

    # Set the number of epochs for training.
    cust_epochs_ = 20

    # Training parameters
    learning_rate_ = 0.01

    # Train dataset
    train_dataset = train_dset()

    # Validation dataset
    validation_dataset = val_dset()

    # Criterion function
    criterion_ = criterion_function()

    # Loaders
    train_loader_, validation_loader_ = loaders()

    # Sigmoid results
    sigmoid_results = train_sigmoid(learning_rate_, input_dim_, hidden_dim1_, hidden_dim2_, output_dim_, criterion_,
                                    train_loader_, validation_loader_, cust_epochs_)

    # Tanh results
    tanh_results = train_tanh(learning_rate_, input_dim_, hidden_dim1_, hidden_dim2_, output_dim_, criterion_,
                              train_loader_, validation_loader_, cust_epochs_)

    # ReLU results
    relu_results = train_relu(learning_rate_, input_dim_, hidden_dim1_, hidden_dim2_, output_dim_, criterion_,
                              train_loader_, validation_loader_, cust_epochs_)

    # Plot training loss
    plot_training_loss(tanh_results, sigmoid_results, relu_results)

    # Plot validation loss
    plot_val_loss(tanh_results, sigmoid_results, relu_results)
