FROM python:3.12-slim

# 1) 시스템 패키지 (필요한 것만)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) 의존성 먼저 복사 -> 캐시 극대화
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 3) 앱 소스 복사
COPY . .

# 4) Render가 런타임에 PORT를 주입하므로, 여기서는 기본값만 지정(옵션)
ENV PORT=8000

# 5) 컨테이너 포트 (문서화용; Render에는 필수 아님)
EXPOSE 8000

# 6) 반드시 0.0.0.0:$PORT 로 바인딩 (JSON exec form에서는 ENV가 치환되지 않으므로 sh -c 사용)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
