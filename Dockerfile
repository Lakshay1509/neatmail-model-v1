FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends build-essential \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

# Python is already present; no curl/wget or paid provider request is needed.
# Coolify uses the image HEALTHCHECK in preference to its dashboard check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8000/openapi.json', timeout=4); assert r.status == 200"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
