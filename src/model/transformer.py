import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Импортируем наш Attention из соседнего файла
from src.model.attention import MultiHeadAttention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Создаем матрицу PE размера (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # Синусы для четных индексов, Косинусы для нечетных
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Добавляем измерение батча: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # register_buffer означает, что это не обучаемый параметр, но его нужно сохранять вместе с весами
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x имеет размер (Batch, Seq_Len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Внимание
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        # Feed-Forward Network (двухслойный персептрон)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        # Нормализация
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Attention + Residual Connection + Norm
        # В оригинальном трансформере norm идет после сложения (Post-LN)
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 2. FFN + Residual Connection + Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int = 3, d_model: int = 128, 
                 num_heads: int = 8, num_layers: int = 3, d_ff: int = 512, 
                 max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Эмбеддинги токенов
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Эмбеддинги позиций
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        # Стек из нескольких Transformer блоков (Encoder)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Финальный классификатор
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (Batch, Seq_Len)
        attention_mask: (Batch, Seq_Len) - единицы для текста, нули для паддинга
        """
        # Получаем эмбеддинги и умножаем на sqrt(d_model) для масштабирования
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Прогоняем через все слои трансформера
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        # x имеет размер (Batch, Seq_Len, d_model)
        # Для классификации всего текста нам нужен один вектор.
        # Обычно берут вектор первого токена [CLS] или усредняют все токены.
        # Поскольку мы используем токенизатор от ruBERT, токен [CLS] стоит на 0-й позиции.
        cls_output = x[:, 0, :]  # Берем вектор только нулевого токена. Размер: (Batch, d_model)
        
        # Прогоняем через полносвязный классификатор
        logits = self.classifier(cls_output)
        return logits

if __name__ == "__main__":
    # Тестируем готовую модель
    vocab_size = 30000  # Примерный размер словаря ruBERT
    model = SentimentTransformer(vocab_size=vocab_size)
    
    # Симуляция батча (Batch=2, Seq_Len=128)
    dummy_input_ids = torch.randint(0, vocab_size, (2, 128))
    dummy_mask = torch.ones(2, 128)
    
    output = model(dummy_input_ids, dummy_mask)
    print("Размерность выходных логитов:", output.shape)
    print("Логиты:\n", output)