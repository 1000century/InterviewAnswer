# results
## 1. Basic Settings
  따로 질문부분 마스킹하지 않았음
  ```python
  def preprocess_training_examples(examples):
    max_length=256
    formatted_inputs = [
        f"</s> 직무: {occupation_map[occ]} 경력상태: {'신입' if exp == 'NEW' else '경력직'} 질문: {q} 답변: {a} </s>"
        for occ, exp, q, a in zip(
            examples["dataSet_info_occupation"],
            examples["dataSet_info_experience"],
            examples["dataSet_question_raw_text"],
            examples["dataSet_answer_raw_text"]
        )
    ]

    tokenized = tokenizer(
        formatted_inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )

    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': [-100 if token == tokenizer.pad_token_id else token for token in tokenized['input_ids']]
    }
  orig_train_dataset = dataset['train'].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=dataset['train'].column_names,
    load_from_cache_file=False  # 이 옵션 추가
  )

  orig_valid_dataset = dataset['validation'].map(
      preprocess_training_examples,
      batched=True,
      remove_columns=dataset['validation'].column_names,
      load_from_cache_file=False  # 이 옵션 추가
  )
  ```

  ```python
  def collate_fn(batch):
     input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
     attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
     labels = torch.stack([torch.tensor(item['labels']) for item in batch])
  
     return {
         'input_ids': input_ids,
         'attention_mask': attention_mask,
         'labels': labels
     }
  
  train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
  valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
  ```
## 2. Results
- ovqa batch(16)_kaggle_pytorch2_v2
  ```python
  def preprocess_training_examples(examples):
    formatted_inputs = [
        f"</s> 직무: {occupation_map[occ]} 경력상태: {'신입' if exp == 'NEW' else '경력직'} 질문: {q} 답변: {a} </s>"
        for occ, exp, q, a in zip(
            examples["dataSet_info_occupation"],
            examples["dataSet_info_experience"],
            examples["dataSet_question_raw_text"],
            examples["dataSet_answer_raw_text"]
        )
    ]

  ```

  ![image](https://github.com/user-attachments/assets/1d84e983-0a4b-411a-b3f1-1bac36798b7d)

- vqa batch(16)_kaggle_pytorch2_v3
  ```python
  def preprocess_training_examples(examples):
    formatted_inputs = [
                f"</s>경력상태: {'신입' if exp == 'NEW' else '경력직'} 질문: {q} 답변: {a} </s>"
        for occ, exp, q, a in zip(
            examples["dataSet_info_occupation"],
            examples["dataSet_info_experience"],
            examples["dataSet_question_raw_text"],
            examples["dataSet_answer_raw_text"]
        )
    ]
  ```
  ![image](https://github.com/user-attachments/assets/b3486cbb-88dd-440f-8282-c6855ed74ca7)

- oqa batch(16)_kagle_pytorch2_v4
  ```python
  def preprocess_training_examples(examples):
    formatted_inputs = [
        f"</s> 직무: {occupation_map[occ]} 질문: {q} 답변: {a} </s>"
        for occ, exp, q, a in zip(
            examples["dataSet_info_occupation"],
            examples["dataSet_info_experience"],
            examples["dataSet_question_raw_text"],
            examples["dataSet_answer_raw_text"]
        )
    ]
  ```
  ![image](https://github.com/user-attachments/assets/b7a3db8a-10be-4e8c-a0ec-ec67a2596a1f)
