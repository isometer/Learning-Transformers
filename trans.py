# %%

from shared_imports import *
from transformer import Transformer
from transformer_config import config


# %%

X_train = np.loadtxt(r"./data/UCI HAR Dataset/train/X_train.txt")
Y_train = np.loadtxt(r"./data/UCI HAR Dataset/train/y_train.txt")

X_test = np.loadtxt(r"./data/UCI HAR Dataset/test/X_test.txt")
Y_test = np.loadtxt(r"./data/UCI HAR Dataset/test/y_test.txt")

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.long)

def create_sliding_windows_tensor(data, labels, win_size=100):
    X, Y = [], []
    for i in range(len(data) - win_size + 1):
        X.append(data[i:i + win_size])
        Y.append(labels[i + win_size - 1])  # Align labels with end of window
    return torch.stack(X), torch.tensor(Y)

X_train, Y_train = create_sliding_windows_tensor(X_train, Y_train, win_size=config.win_size)
X_test, Y_test = create_sliding_windows_tensor(X_test, Y_test, win_size=config.win_size)

# Create DataLoader for training data
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

# Create DataLoader for test data (for evaluation purposes)
test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

Y_train -= 1  # Our labels are 1-indexed, adjust for 0-indexing downstream
Y_test -= 1


# %%

def my_plot(epochs, loss):
    plt.plot(epochs, loss)

# define criterion for training
criterion = nn.CrossEntropyLoss()

transformer = Transformer(config.input_c, config.output_c, config.d_model, config.k, 
                          config.num_layers, config.d_ff, config.win_size, config.dropout)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train() # puts the model in 'training mode', allowing dropout etc.

loss_vals = []
for epoch in range(config.num_epochs):
    epoch_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        # print(f"Input shape: {inputs.shape}")

        # Forward pass
        output = transformer(inputs)

        if output.shape[1] != config.output_c:
            raise ValueError(f"Expected output shape {(config.batch_size, config.output_c)} but got {output.shape}")

        # Calculate loss
        loss = criterion(output, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Epoch: {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")
# %%
transformer.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, targets in test_loader:
        outputs = transformer(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# %%
