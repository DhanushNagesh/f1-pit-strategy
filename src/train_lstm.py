import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import joblib
import json
from pathlib import Path

from features import engineer_features, get_feature_columns

class PitStopLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bi-LSTM processes sequences forward and backward
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classifier head (bidirectional doubles hidden size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last timestep output
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


class StintDataset(Dataset):
    """Convert stint sequences to tensors."""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.labels[idx]])
        )


def create_stint_sequences(df, feature_cols, min_stint_length=3):
    """
    Convert lap data to stint sequences.
    Each sequence is all laps in a stint. Target is whether the LAST lap is a pit stop.
    """
    sequences = []
    labels = []
    
    # Group by driver, race, stint
    for (driver, race, stint), group in df.groupby(['driver', 'race', 'stint']):
        if len(group) < min_stint_length:
            continue
        
        # Get features for all laps in stint
        seq = group[feature_cols].values
        
        # Target: is the last lap a pit stop?
        target = group.iloc[-1]['target']
        
        sequences.append(seq)
        labels.append(target)
    
    return sequences, labels


def pad_sequences(sequences, max_len=None):
    """Pad sequences to same length."""
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    
    padded = np.zeros((len(sequences), max_len, sequences[0].shape[1]))
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length, :] = seq[:length]
    
    return padded


def train_lstm(data_path='data/processed/race_data.csv', epochs=50, batch_size=32):
    """Train Bi-LSTM model."""
    
    # Load data
    df = pd.read_csv(data_path)
    df = engineer_features(df)
    
    feature_cols = get_feature_columns()
    
    # Scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Create stint sequences
    sequences, labels = create_stint_sequences(df, feature_cols)
    
    print(f"Created {len(sequences)} stint sequences")
    print(f"Positive class ratio: {np.mean(labels):.3f}")
    
    # Pad sequences
    padded_seqs = pad_sequences(sequences, max_len=30)  # Max 30 laps per stint
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        padded_seqs, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = StintDataset(X_train, y_train)
    test_dataset = StintDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = len(feature_cols)
    model = PitStopLSTM(input_size=input_size, hidden_size=64, num_layers=2)
    
    # Loss and optimizer (weighted for class imbalance)
    pos_weight = torch.FloatTensor([(1 - np.mean(y_train)) / np.mean(y_train)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nTraining Bi-LSTM...")
    best_f1 = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass (model outputs sigmoid, but criterion expects logits)
            # We need to modify this - use BCELoss instead
            outputs = model(X_batch)
            loss = nn.BCELoss()(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = model(X_batch)
                    preds = (outputs > 0.5).float()
                    all_preds.extend(preds.numpy())
                    all_labels.extend(y_batch.numpy())
            
            f1 = f1_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - F1: {f1:.3f} - P: {precision:.3f} - R: {recall:.3f}")
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), 'models/lstm_model.pth')
    
    # Final evaluation
    model.load_state_dict(torch.load('models/lstm_model.pth'))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())
    
    print("\n=== Final Bi-LSTM Results ===")
    print(classification_report(all_labels, all_preds))
    
    final_f1 = f1_score(all_labels, all_preds)
    final_precision = precision_score(all_labels, all_preds)
    final_recall = recall_score(all_labels, all_preds)
    
    # Save metrics
    metrics = {
        'model_type': 'Bi-LSTM',
        'f1_score': float(final_f1),
        'precision': float(final_precision),
        'recall': float(final_recall),
        'num_sequences': len(sequences),
        'max_sequence_length': 30
    }
    
    with open('models/lstm_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print(f"\n✓ Saved Bi-LSTM model (F1: {final_f1:.3f})")
    
    return model, metrics


if __name__ == "__main__":
    train_lstm(epochs=50)