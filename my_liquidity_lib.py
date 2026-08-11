
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import cvxpy as cp

def build_temporal_sequences(features, lookback):
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i+lookback, :])
        y.append(features[i+lookback, 0])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

class SequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, lookback_window=12):
        super(SequenceModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * lookback_window, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

def train_sequence_model(model, X_train, y_train, epochs=150, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model

def llm_knowledge_extractor(month_index):
    """
    Simulates the LLM parsing unstructured boardroom texts into KG entities.
    This function specifically returns the C_max limit based on month_index.
    """
    if month_index < 12:
        return 1.0
    else:
        return 0.5

def calculate_static_metrics(actual, predicted, horizons, c_max_limit):
    metrics = []
    for h in horizons:
        act_h = actual[:h]
        pred_h = predicted[:h]
        rmse = np.sqrt(np.mean((act_h - pred_h)**2))
        violations = np.sum(pred_h > c_max_limit)
        avr = (violations / h) * 100
        metrics.extend([rmse, avr])
    return metrics

def calculate_dynamic_metrics(actual, predicted, dynamic_limits, horizons):
    metrics = []
    for h in horizons:
        act_h = actual[:h]
        pred_h = predicted[:h]
        dyn_lim_h = dynamic_limits[:h]

        rmse = np.sqrt(np.mean((act_h - pred_h)**2))

        violations = 0
        for i in range(h):
            if pred_h[i] > dyn_lim_h[i] + 1e-5: # adding small epsilon for float math
                violations += 1
        avr = (violations / h) * 100
        metrics.extend([rmse, avr])
    return metrics

def train_and_predict_sequence_model(df_data, shock_t, lookback, input_dim, feature_columns, epochs=150, target_col_idx=0):
    scaler = StandardScaler()
    features_data = df_data[feature_columns].values
    train_scaled = scaler.fit_transform(features_data[:shock_t])

    X_train, y_train = build_temporal_sequences(train_scaled, lookback)

    torch.manual_seed(42) # For reproducibility
    model = SequenceModel(input_dim=input_dim, lookback_window=lookback)
    model = train_sequence_model(model, X_train, y_train, epochs=epochs)

    # Test set inference
    test_scaled = scaler.transform(features_data[shock_t - lookback : shock_t + 36 - 1])
    X_test = torch.tensor(np.array([test_scaled[i:i+lookback, :] for i in range(36)]), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        scaled_pred = model(X_test).numpy().squeeze()

    # Inverse transform
    dummy = np.zeros((len(scaled_pred), input_dim))
    dummy[:, target_col_idx] = scaled_pred
    forecast = scaler.inverse_transform(dummy)[:, target_col_idx]

    return forecast
