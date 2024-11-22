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
```python
train, valid batch = 32,32
```
3. 학습 파라미터 설정
```python
num_epochs = 5
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


# validation에서 벌어지는 일
- Reference
- 직무: 영업마케팅 경력상태: 신입 질문: 금융정책기관으로서 신용보증기금이 허가 돼야 할 업무 영역은 무엇이라고 **생각하는지 답변:** 신용보증기금은 신용보증을 통해서 중소기업의 금융을 원활히 하고 어 국민 경제 발전에 기여하는 목적을 지닌 그런 금융정책기관이라고 할 수 있습니다. 어 중소기업적인 어 그런 금융기관인 만큼 보다 서민적으로 제 제도적인 어떤 지원을 해줄 수 있는 방편을 매해 알아보고 어 보다 폭넓은 예산을 지원받아 어 보다 친근하고 어 실 실속 있는 그런 지원이 약속되는 어 그런 어 그런 어떤 금융기관이 되어야 할 것으로 생각됩니다. 예. 
- Predicted
- 직무: 연구개발 경력 경력상태: 신입 질문: 지원권에서 관련 금융평가기금과 가지고받은서 하는 부분이 분야가 무엇이라고 **생각지 궁금변:** 저보증기금은 금융보증증을 할 대출에 자금을 지원 지원 그 소경제의 생활을 이바지 기관이 가지고 기관 기관이기관으로기관입니다. 생각 수 있습니다.  저에 입장에서 그런 입장에서을 것 어 더과 어 역할을권 부분을 지원을 해드 수 있는 그런안이 강구뉴얼 강구보고 있습니다. 그 더 시각에서 지원해서 중소기업 서민하게 따뜻한 중소기업리속 있는 그런 금융 되도록된 그런 그런 기관이 그런 금융 그런 되기를 된다고 것 생각합니다.니다. . 저
- Reference와 Predicted를 교차되면서 확인해보면, **생각** > 지, **하는지** > 궁금, **답** > 변 이런식으로 밀려서 예측되는 것이 확인됨
