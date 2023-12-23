from model import PredictionModel
from dataset import NearDataset

import torch
import argparse

# This is where training would happen
if __name__ == "__main__":
    # Read the args
    parser = argparse.ArgumentParser(description="Train a model on NEAR dataset")
    parser.add_argument("--train", type=str, default="lstm", help="The model to train")

    args = parser.parse_args()

    # Initialize the dataset
    near_dataset = NearDataset()

    # TODO: Tune hypermeters
    # Hyperparameters
    input_size = len(near_dataset.features)
    hidden_size = 128
    num_layers = 3
    output_size = 1
    batch_size = 64
    num_epochs = 100

    # Initialize the model
    model = PredictionModel(input_size, hidden_size, num_layers, output_size)

    # Train the model
    loss_function = torch.nn.MSELoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Split the dataset into two
    train_dataset, test_dataset = near_dataset.train_test_split()

    # Initialize the data loaders
    # Shuffle doesn't matter because we are using LSTM
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    if int(args.train) == 1:
        # Train the model
        for epoch in range(num_epochs):
            # Forward pass
            for sequence, label in train_data_loader:
                # Zero out the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(sequence.float())

                # Calculate the loss
                loss = loss_function(outputs, label.float())

                # Backward and optimize
                loss.backward()
                optimizer.step()

            # Print the loss
            print(f"Epoch {epoch} Loss: {loss.item()}")

        # Save the model
        torch.save(model.state_dict(), "./model_training/models/model.pth")
    
    if int(args.train) == 0:
        model.load_state_dict(torch.load("./model_training/models/model.pth"))

    # Test the model on the test dataset
    model.eval()
    total_test_loss = 0
    total_test_samples = len(test_data_loader.dataset)


    with torch.no_grad():
        for sequence, label in test_data_loader:
            outputs = model(sequence.float())
            loss = loss_function(outputs, label.float())
            total_test_loss += loss.item() * sequence.size(0)
            labels = label
        
    # Calculate the average loss
    average_test_loss = total_test_loss / total_test_samples
    print(f"Average Test Loss: {average_test_loss:.4f}")


    # Do one pass on the test dataset
    for sequence, label in test_data_loader:
        outputs = model(sequence.float())

        # Take these outputs and turn them into a prection
        prediction_output = near_dataset.target_scaler.inverse_transform(outputs.detach().numpy())
        real_output = near_dataset.target_scaler.inverse_transform(label.detach().numpy().reshape(-1,1))
        
        # Compare 1 to 1
        for i in range(len(prediction_output)):
            print(f"Predicted: {prediction_output[i][0]}, Real: {real_output[i][0]}")


# TODO: Testing
# Ensure that the data arrives sequentially, so that the LSTM could work
# Ensure that the data is normalized
# Improve the dataset with the BTC price