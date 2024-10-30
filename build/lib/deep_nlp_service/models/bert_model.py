import torch
import torch.nn as nn
from transformers import BertModel
from .attention import MultiHeadAttention

class EnhancedDeepNLPModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, 
                 n_layers: int, dropout: float, num_heads: int = 8):
        super().__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0, 
            batch_first=True
        )
        
        self.gru = nn.GRU(
            hidden_dim * 2, 
            hidden_dim, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True
        )
        
        self.attention = MultiHeadAttention(hidden_dim * 2, num_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 2)
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]
        bert_output = self.bert_dropout(bert_output)
        
        lstm_output, (hidden, cell) = self.lstm(bert_output)
        lstm_output = self.layer_norm1(lstm_output)
        
        gru_output, hidden = self.gru(lstm_output)
        gru_output = self.layer_norm2(gru_output + lstm_output)
        
        attention_output = self.attention(gru_output, gru_output, gru_output, attention_mask)
        pooled_output = torch.mean(attention_output, dim=1)
        
        dense1 = self.activation(self.fc1(pooled_output))
        dense1 = self.dropout(dense1)
        
        dense2 = self.activation(self.fc2(dense1))
        dense2 = self.dropout(dense2)
        
        return self.fc3(dense2)
