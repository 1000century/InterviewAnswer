from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# 모델 및 토크나이저 설정
model_dir  = 'C:\\Users\\karen\\chatbot\\chatbot\\model\\'  # 파인튜닝된 모델 파일 경로
# model_path = "model/"  # Docker 컨테이너 내부의 경로
# Best Model 로드
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 모델의 상태 설정
device = torch.device('cpu')  # 'cuda'를 사용할 수 있는 경우 'cuda'로 변경
model.to(device)
model.eval()

# 응답 생성 함수
def generate_response(prompt, max_length=250):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids,
        max_length=max_length,
        no_repeat_ngram_size=2,
        num_beams=5,
        temperature=0.7,
        early_stopping=True,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    if "답변:" in response:
        response = response.split("답변:")[-1].strip()

    if prompt in response:
        response = response.replace(prompt, "").strip()

    response = response.replace("\n", " ").strip()
    
    return response

# 홈 페이지 (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# 챗봇 페이지 (chat.html)
@app.route('/chat')
def chat():
    return render_template('chat.html')

# 챗봇 응답 API 엔드포인트
@app.route('/generate_response', methods=['POST'])
def chatbot_response():
    user_input = request.json['user_input']  # JavaScript에서 전달한 사용자 질문
    response = generate_response(user_input)  # 생성된 응답
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
