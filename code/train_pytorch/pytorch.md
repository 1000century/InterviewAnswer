허깅페이스 trainer 없이 학습
# 파라미터
1. Dataset 구성
```python
tokenizer max_length=256

input_text = f" 질문: {question} 경력: {veteran} 직업: {occupation} 답변: {answer} "
encoding = self.tokenizer(
    input_text,
    max_length=self.max_length, # 256
    padding='max_length',
    truncation=True,
    return_tensors="pt"
)

input_ids = encoding['input_ids'].squeeze()
attention_mask = encoding['attention_mask'].squeeze()

# labels를 input_ids의 복사본
labels = input_ids.clone()

# 패딩 토큰 인덱스를 -100
labels[input_ids == self.tokenizer.pad_token_id] = -100

return {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'labels': labels
}
```
2. DataLoader 구성

train, valid batch = 32,32

3. 학습 파라미터 설정
num_epochs = 5

```python
optimizer = AdamW(model.parameters(), lr=5e-5)  # 옵티마이저 정의
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)  # 스케줄러 정의
```

# 결과
- Average Validation Loss: 3.0023
- Average Validation BLEU: 1.7672
- Average Validation ROUGE: {'rouge1': 0.3065894424121382, 'rouge2': 0.09108117509414317, 'rougeL': 0.2421079120163181}
- BERTScore Precision: 0.7213
- BERTScore Recall: 0.7598
- BERTScore F1: 0.7390
