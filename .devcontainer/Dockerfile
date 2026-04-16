FROM python:3.11-slim

WORKDIR /workspaces

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

CMD ["sleep", "infinity"]
