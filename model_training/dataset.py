import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class NearDataset(torch.utils.data.Dataset):

    def __init__(self, directory="./data/", sequence_length=10, filename="NEAR_USDT_pricedata_processed.csv", target_column="near_nexthourprice"):
        self.sequence_length = sequence_length # This is only applicable if you are using LSTM
        self.target_column = target_column
        self.features = ["price_open", "price_high", "price_low", "price_close", "volume_traded", "trades_count", "year", "day_of_week", "hour_of_day"]
        self.df = pd.read_csv(directory + filename)  
        self._preprocessing_data(target_column)
        self._create_sequences()

    
    def _preprocessing_data(self, target_column):
        self.df["time_open"] = pd.to_datetime(self.df["time_open"])
        self.df["year"] = self.df["time_open"].dt.year
        self.df["day_of_week"] = self.df["time_open"].dt.dayofweek
        self.df["hour_of_day"] = self.df["time_open"].dt.hour

        features = self.features
        target = [target_column]

        self.feature_scaler = MinMaxScaler()
        self.df[features] = self.feature_scaler.fit_transform(self.df[features])

        self.target_scaler = MinMaxScaler()
        self.df[target] = self.target_scaler.fit_transform(self.df[target])

        self.df = self.df.drop(["time_open", "time_close", "time_period_start", "time_period_end", "near_nextdayprice", "near_nextweekprice"], axis=1)
    
    def _create_sequences(self):
        sequences = []
        data_size = len(self.df) 
        for i in range(data_size - self.sequence_length):
            sequence = self.df.iloc[i:i+self.sequence_length][self.features]
            label = self.df.iloc[i+self.sequence_length][self.target_column]
            sequences.append((sequence, label))
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return torch.tensor(sequence.values), torch.tensor(label)
    
# Test the dataset
if __name__ == "__main__":
    # Initialize the dataset
    near_dataset = NearDataset()

    # Check initial few rows after preprocessing
    print("Processed Data:")
    print(near_dataset.df.head())

    # Check length of the dataset
    print("Total Sequences:", len(near_dataset))

    # Inspect a few sequences
    for i in range(2):
        sequence, label = near_dataset[i]
        print(f"Sequence {i} Shape: {sequence.shape}, Label: {label}")

    # Test with DataLoader
    data_loader = DataLoader(near_dataset, batch_size=32, shuffle=True)
    for batch in data_loader:
        sequences, labels = batch
        print("Batch of Sequences Shape:", sequences.shape)
        print("Batch of Labels Shape:", labels.shape)
        break  # Just to check the first batch