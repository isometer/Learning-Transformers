# %%

from shared_imports import *
from transformer import Transformer
from transformer_config import config
from wfdb_reader import *


# %%

X_data = WfdbReader().fetch_data()
X_data = torch.tensor(X_data, dtype=torch.float32) # dimensions: (examples, samples, features)

def create_sliding_windows_tensor(data, win_size=100, step=1):
    X = []
    indices = []
    for i in range(0, data.shape[1] - win_size + 1, step):
        X.append(data[:, i:i + win_size, :])  # Create sliding window
        indices.append(range(i, i + win_size))  # Save the indices of this window
    return torch.cat(X, dim=0), indices

X_data, indices = create_sliding_windows_tensor(X_data, win_size=config.win_size, step=25)

print(f"data size: {X_data.shape}")

# Create DataLoader for training data
dataset = TensorDataset(X_data)
loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# %%

def my_plot(epochs, loss):
    plt.plot(epochs, loss)

# define criterion for training
criterion = nn.MSELoss()

using_cuda = torch.cuda.is_available()

transformer = Transformer(config.input_c, config.output_c, config.d_model, config.k, 
                          config.num_layers, config.d_ff, config.win_size, config.dropout)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer.to(device)

optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train() # puts the model in 'training mode', allowing dropout etc.

# the suspected problem: sliding windows at intervals of only one reading
# creates a shit ton of sliding windows. Try spreading these out more.

loss_vals = []
for epoch in range(config.num_epochs):
    epoch_loss = 0
    for batch_idx, inputs in enumerate(loader):
        inputs = torch.stack(inputs).squeeze(0) # removes first dimension of size 1        
        optimizer.zero_grad()

        targets = inputs # critical: we are doing RECONSTRUCTION

        inputs, targets = inputs.to(device), targets.to(device)

        output = transformer(inputs)

        # Calculate loss
        loss = criterion(output, targets)

        if(batch_idx % 100 == 0):
            print(f"batch index {batch_idx}, loss: {loss}")

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Epoch: {epoch + 1}, Loss: {epoch_loss / len(loader)}")
# %%
transformer.eval()
with torch.no_grad():
    # EVALUATE SOMEHOW
    print("evaluation is a later problem...")
    # Initialize a zero array for the original data
    original_length = X_data.shape[1]
    anomaly_scores = np.zeros((original_length,))

    predictions = [] # fill in later, do a testing split
    threshold = 1 # fix later when I know what these look like

    # Aggregate predictions back to the original data
    for i, idx_range in enumerate(indices):
        for idx in idx_range:
            anomaly_scores[idx] += predictions[i]  # Accumulate predictions

    # Optionally normalize or threshold
    anomaly_scores /= len(indices)
    anomaly_labels = (anomaly_scores > threshold).astype(int)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(original_data, label="Original Data")
    plt.plot(anomaly_scores, label="Anomaly Scores", color="red")
    plt.legend()
    plt.show()
    

# %%
