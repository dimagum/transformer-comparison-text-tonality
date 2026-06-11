import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys

# Добавляем корень в пути, чтобы импорты работали даже без $env:PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.transformer import SentimentTransformer
from src.data_pipeline.dataset import get_dataloaders
import mlflow

def train_model(epochs=3, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # Загружаем датасет
    train_loader, val_loader, tokenizer = get_dataloaders(batch_size=batch_size)
    vocab_size = tokenizer.vocab_size

    # Инициализируем нашу модель
    model = SentimentTransformer(vocab_size=vocab_size).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Настраиваем MLflow локально
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Custom_Transformer_Sentiment")

    with mlflow.start_run():
        # Логируем параметры
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr, "d_model": model.d_model})

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            # Обертка tqdm для красивого прогресс-бара
            progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{epochs}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                
                # Подсчет точности (Accuracy)
                predictions = torch.argmax(outputs, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total
            
            print(f"Эпоха {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
            
            # Логируем метрики в MLflow
            mlflow.log_metrics({"train_loss": avg_loss, "train_accuracy": accuracy}, step=epoch)

        # Сохраняем веса модели
        torch.save(model.state_dict(), "custom_transformer.pt")
        mlflow.log_artifact("custom_transformer.pt")
        print("Обучение завершено. Веса сохранены.")

if __name__ == "__main__":
    # На RTX 5070 Ti можно поставить batch_size побольше (например, 64 или 128)
    train_model(epochs=3, batch_size=64, lr=1e-4)