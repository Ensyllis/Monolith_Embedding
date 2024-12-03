FROM python:3.8-slim

# Install system dependencies including Rust
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pipeline.py .

# Create data directories
RUN mkdir -p data/input data/processed data/embedding_results

CMD ["python", "pipeline.py"]