#coding:utf8
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

from transformers import BertModel, BertTokenizer

"""
基于pytorch的BERT语言模型
"""

class CustomLanguageModel(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(CustomLanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese', return_dict=False)
        self.fc = nn.Linear(input_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        attention_mask = torch.ones(x.size(), device=x.device)
        attention_mask = torch.triu(attention_mask).transpose(0, 1)
        bert_output = self.bert(x, attention_mask=attention_mask)[0]
        logits = self.fc(bert_output)
        
        if y is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            return loss
        else:
            return torch.softmax(logits, dim=-1)

def load_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for idx, line in enumerate(f):
            char = line.strip()
            vocab[char] = idx + 1  # 0 is reserved for <pad>
    return vocab

def load_corpus(corpus_path):
    with open(corpus_path, encoding="gbk") as f:
        corpus = f.read().replace('\n', '')
    return corpus

def create_sample(vocab, window_size, corpus, tokenizer):
    start_idx = random.randint(0, len(corpus) - window_size - 1)
    input_seq = corpus[start_idx:start_idx + window_size]
    target_seq = corpus[start_idx + 1:start_idx + window_size + 1]
    
    x = tokenizer.encode(input_seq, add_special_tokens=False, padding='max_length', truncation=True, max_length=window_size)
    y = tokenizer.encode(target_seq, add_special_tokens=False, padding='max_length', truncation=True, max_length=window_size)
    
    return torch.tensor(x), torch.tensor(y)

def build_dataset(num_samples, vocab, window_size, corpus, tokenizer):
    inputs, targets = [], []
    for _ in range(num_samples):
        x, y = create_sample(vocab, window_size, corpus, tokenizer)
        inputs.append(x)
        targets.append(y)
    return torch.stack(inputs), torch.stack(targets)

def create_model(vocab, char_dim):
    model = CustomLanguageModel(char_dim, len(vocab))
    return model

def generate_text(prompt, model, vocab, tokenizer, window_size):
    model.eval()
    generated_text = prompt
    with torch.no_grad():
        while len(generated_text) < 30 and generated_text[-1] != '\n':
            x = tokenizer.encode(generated_text[-window_size:], add_special_tokens=False, padding='max_length', truncation=True, max_length=window_size)
            x = torch.tensor(x).unsqueeze(0).to(next(model.parameters()).device)
            logits = model(x)
            next_token_id = sample_from_logits(logits[0, -1])
            next_token = tokenizer.decode([next_token_id])
            generated_text += next_token
    return generated_text

def sample_from_logits(logits):
    if random.random() > 0.2:
        return logits.argmax().item()
    else:
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        return np.random.choice(len(probabilities), p=probabilities)

def calculate_perplexity(sentence, model, vocab, tokenizer, window_size):
    model.eval()
    log_prob_sum = 0.0
    with torch.no_grad():
        for i in range(1, len(sentence)):
            x = tokenizer.encode(sentence[max(0, i - window_size):i], add_special_tokens=False, padding='max_length', truncation=True, max_length=window_size)
            x = torch.tensor(x).unsqueeze(0).to(next(model.parameters()).device)
            logits = model(x)
            target_token = sentence[i]
            target_idx = vocab.get(target_token, vocab["<UNK>"])
            log_prob_sum += math.log(torch.softmax(logits[0, -1], dim=-1)[target_idx].item())
    return math.exp(-log_prob_sum / len(sentence))

def train_model(corpus_path, save_model=True):
    num_epochs = 15
    batch_size = 32
    num_samples = 40000
    char_dim = 256
    window_size = 10

    vocab = load_vocab("vocab.txt")
    corpus = load_corpus(corpus_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    model = create_model(vocab, char_dim)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    print("Training started...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for _ in range(num_samples // batch_size):
            x_batch, y_batch = build_dataset(batch_size, vocab, window_size, corpus, tokenizer)
            if torch.cuda.is_available():
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            optimizer.zero_grad()
            loss = model(x_batch, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / (num_samples // batch_size)
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        print(generate_text("天子笑时人间万里", model, vocab, tokenizer, window_size))
        print(generate_text("风吹柳絮雪漫天", model, vocab, tokenizer, window_size))

    if save_model:
        model_save_path = os.path.join("model", os.path.basename(corpus_path).replace(".txt", ".pth"))
        torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    train_model("corpus.txt", save_model=False)