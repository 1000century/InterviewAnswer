# Stage 1: 빌드 환경
FROM python:3.12-slim as builder

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y \
    build-essential gcc \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: 런타임 환경
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 빌드 단계에서 설치된 Python 의존성 복사
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12

# 소스 코드 복사
COPY . /app

# 환경 변수 설정
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# 모델 파일은 외부에서 마운트하거나 다운로드
VOLUME /app/model

# Flask 애플리케이션 실행
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]

