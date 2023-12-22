import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


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
    

if __name__ == "__main__":
    # Initialize a NEAR dataset
    near_dataset = NearDataset()

    print(near_dataset.df.head())
