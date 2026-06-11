import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class SentimentDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.texts = hf_dataset['text']
        self.labels = hf_dataset['label']
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Токенизация текста
        encoding = self.tokenizer(
            text,
            add_special_tokens=True, # Добавляет [CLS] и [SEP]
            max_length=self.max_length,
            padding='max_length',    # Дополняет нулями до max_length
            truncation=True,         # Обрезает длинные тексты
            return_tensors='pt'      # Возвращает PyTorch тензоры
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_dataloaders(batch_size=32, max_length=128):
    """
    Загружает датасет с HF, токенизирует и возвращает DataLoader'ы
    """
    print("Загрузка датасета MonoHime/ru_sentiment_dataset...")
    # Датасет загружается напрямую из Hugging Face Hub
    dataset = load_dataset("MonoHime/ru_sentiment_dataset")
    
    # Так как датасет может не иметь валидационной выборки изначально, разобьем train
    split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
    
    print("Инициализация токенизатора rubert-tiny2...")
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")

    train_dataset = SentimentDataset(split_dataset['train'], tokenizer, max_length)
    val_dataset = SentimentDataset(split_dataset['test'], tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer

if __name__ == "__main__":
    # Быстрый тест в рамках ветки feat/data-prep
    train_loader, val_loader, tokenizer = get_dataloaders(batch_size=4)
    
    batch = next(iter(train_loader))
    print("Размерность input_ids:", batch['input_ids'].shape)
    print("Размерность attention_mask:", batch['attention_mask'].shape)
    print("Метки классов:", batch['labels'])