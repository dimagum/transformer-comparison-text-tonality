import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Инициализация Multi-Head Attention.
        d_model: размерность входных эмбеддингов (например, 128)
        num_heads: количество голов (например, 8)
        """
        super().__init__()
        
        # Проверяем, что размерность делится на количество голов без остатка
        assert d_model % num_heads == 0, "d_model должно делиться на num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Размерность одной головы
        
        # Линейные проекции для Query, Key и Value
        # Вместо трех отдельных слоев мы можем использовать один, который выдает размер 3 * d_model
        # Это эффективнее для параллельных вычислений на GPU
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        
        # Финальная линейная проекция
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: тензор размерности (Batch, Seq_Len, d_model)
        mask: маска для игнорирования padding-токенов
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. Проецируем вход x на Q, K, V одним умножением матрицы
        # Размер: (Batch, Seq_Len, 3 * d_model)
        qkv = self.qkv_proj(x)
        
        # Разбиваем на Query, Key, Value
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 2. Меняем размерность для разделения на головы (heads)
        # Исходный размер: (Batch, Seq_Len, d_model)
        # Новый размер: (Batch, Seq_Len, num_heads, d_k)
        # После transpose: (Batch, num_heads, Seq_Len, d_k)
        # Это нужно, чтобы batch-умножение матриц работало независимо для каждой головы
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Вычисляем Scaled Dot-Product Attention
        # Умножаем Q на K^T
        # Размер scores: (Batch, num_heads, Seq_Len, Seq_Len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Применяем маску (заменяем 0 на очень маленькое число, чтобы после softmax стал 0)
        # Маска нужна, чтобы не обращать внимание на [PAD] токены
        if mask is not None:
            # Расширяем маску для голов: (Batch, 1, 1, Seq_Len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Применяем softmax для получения вероятностей (attention weights)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Умножаем веса на Value
        # Размер output: (Batch, num_heads, Seq_Len, d_k)
        output = torch.matmul(attn_weights, v)
        
        # 4. Конкатенируем головы обратно
        # Сначала возвращаем transpose назад: (Batch, Seq_Len, num_heads, d_k)
        # Затем применяем contiguous() и view() для склеивания в (Batch, Seq_Len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 5. Финальная проекция
        return self.out_proj(output)

if __name__ == "__main__":
    # Тестируем работоспособность слоя
    batch_size, seq_len, d_model = 2, 128, 256
    num_heads = 8
    
    # Симулируем батч с эмбеддингами
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Симулируем маску, где некоторые токены в конце - padding (нули)
    mask = torch.ones(batch_size, seq_len)
    mask[:, -10:] = 0 
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    output = mha(x, mask)
    print(f"Размерность входа: {x.shape}")
    print(f"Размерность выхода: {output.shape}")