
import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_layers=2, dropout=0.1):
        super(MLPModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 定义 MLP 层
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # 组合成 Sequential
        self.model = nn.Sequential(*layers)
        
        # 初始化
        self._init_weights()

    def _init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        # x: [batch_size, n_cells, input_dim]
        return self.model(x)  # [batch_size, n_cells, output_dim]