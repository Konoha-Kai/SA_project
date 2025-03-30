import torch
import torch.nn as nn
import torch.optim as optim
from transformer import TransformerModel
from mlp import MLPModel
import numpy as np

# 取消警告
import warnings
warnings.filterwarnings("ignore")

# 超参数
class Config:
    n_cells = 1000         # 总细胞数
    input_dim = 50        # 每个细胞的特征数
    output_dim = 200      # 每个细胞的基因数
    batch_size = 32
    epochs = 10
    model_type = "mlp"  # 可切换为 "mlp"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformer 参数
    n_heads = 4
    n_layers = 2
    hidden_dim = 128
    dropout = 0.1

    # 训练参数
    lr = 0.001
    group_size = 100      # 每 100 个细胞为一组

# 数据模拟
def generate_data(config):
    # 模拟输入数据: [n_cells, input_dim]
    X = torch.randn(config.n_cells, config.input_dim)
    
    # 模拟部分真实基因数据: 只知道每 100 个细胞的基因和
    n_groups = config.n_cells // config.group_size
    y_true_grouped = torch.randn(n_groups, config.output_dim)  # [n_groups, output_dim]
    
    # 模拟完整的基因数据（仅用于验证）
    y_true = torch.randn(config.n_cells, config.output_dim)
    
    return X, y_true, y_true_grouped

# 数据加载器
def get_dataloader(X, batch_size):
    dataset = torch.utils.data.TensorDataset(X)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 自定义损失函数（包含分组和约束）
def compute_loss(pred, y_true_grouped, group_size):
    # pred: [batch_size, n_cells, output_dim]
    batch_size = pred.size(0)
    n_cells = pred.size(1)
    n_groups = n_cells // group_size
    
    # 将预测按组求和
    pred_grouped = pred.view(batch_size, n_groups, group_size, -1).sum(dim=2)  # [batch_size, n_groups, output_dim]
    
    # 计算分组数据的损失
    loss_grouped = nn.MSELoss()(pred_grouped, y_true_grouped)
    
    # 可添加其他正则化项（如预测值的平滑性）
    return loss_grouped

# 训练函数
def train_model(model, dataloader, y_true_grouped, config):
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    model.train()
    
    for epoch in range(config.epochs):
        total_loss = 0
        for batch in dataloader:
            X_batch = batch[0].to(config.device)  # [batch_size, n_cells, input_dim]
            X_batch = X_batch.unsqueeze(1).repeat(1, config.n_cells, 1)  # 模拟全细胞输入
            
            optimizer.zero_grad()
            pred = model(X_batch)  # [batch_size, n_cells, output_dim]
            
            # 计算损失
            loss = compute_loss(pred, y_true_grouped.to(config.device), config.group_size)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {total_loss / len(dataloader):.4f}")

# 主函数
def main():
    config = Config()
    
    # 生成数据
    X, y_true, y_true_grouped = generate_data(config)
    dataloader = get_dataloader(X, config.batch_size)
    
    # 选择模型
    if config.model_type == "transformer":
        model = TransformerModel(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        ).to(config.device)
    elif config.model_type == "mlp":
        model = MLPModel(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout=config.dropout
        ).to(config.device)
    else:
        raise ValueError("Unknown model type")

    # 训练模型
    train_model(model, dataloader, y_true_grouped, config)
    
    # 测试推理
    model.eval()
    with torch.no_grad():
        X_test = X[:config.batch_size].to(config.device)
        X_test = X_test.unsqueeze(1).repeat(1, config.n_cells, 1)
        pred = model(X_test)
        print("Sample prediction shape:", pred.shape)

if __name__ == "__main__":
    main()