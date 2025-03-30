import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads=4, n_layers=2, hidden_dim=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 输入投影层
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出层
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # 初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, x):
        # x: [batch_size, n_cells, input_dim]
        x = self.input_proj(x)  # [batch_size, n_cells, hidden_dim]
        x = self.transformer(x)  # [batch_size, n_cells, hidden_dim]
        x = self.output_proj(x)  # [batch_size, n_cells, output_dim]
        return x