import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Tuple
import json
from datetime import datetime
import shutil
from tqdm import tqdm

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("StylometryPipeline")
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = RotatingFileHandler(log_dir / 'pipeline.log', maxBytes=10*1024*1024, backupCount=5)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch) 
    
    return logger

def chunk_text(text: str, tokenizer, chunk_size: int = 512) -> list:
    """Split text into chunks based on token count, handling long sequences safely."""
    chunks = []
    rough_chunks = [p for p in text.split('. ') if p.strip()]
    
    current_chunk = []
    current_length = 0
    
    for rough_chunk in tqdm(rough_chunks, desc="Processing text chunks", leave=False):
        try:
            tokens = tokenizer.encode(rough_chunk + '.', add_special_tokens=False)
        except Exception as e:
            logger.warning(f"Error tokenizing chunk, splitting further: {str(e)}")
            words = rough_chunk.split()
            tokens = []
            current_piece = []
            
            for word in words:
                current_piece.append(word)
                if len(' '.join(current_piece)) > 200: 
                    try:
                        piece_tokens = tokenizer.encode(' '.join(current_piece) + '.', add_special_tokens=False)
                        tokens.extend(piece_tokens)
                        current_piece = []
                    except Exception as e:
                        logger.error(f"Could not tokenize piece: {str(e)}")
                        current_piece = []
                        continue
            
            if current_piece:
                try:
                    piece_tokens = tokenizer.encode(' '.join(current_piece) + '.', add_special_tokens=False)
                    tokens.extend(piece_tokens)
                except Exception as e:
                    logger.error(f"Could not tokenize final piece: {str(e)}")
        
        if current_length + len(tokens) > chunk_size and current_chunk:
            chunk_text = tokenizer.decode(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            current_chunk = []
            current_length = 0
        
        if len(tokens) > chunk_size:
            for i in range(0, len(tokens), chunk_size):
                sub_tokens = tokens[i:i + chunk_size]
                sub_text = tokenizer.decode(sub_tokens)
                if sub_text.strip():
                    chunks.append(sub_text.strip())
        else:
            current_chunk.extend(tokens)
            current_length += len(tokens)
        
        if current_length >= chunk_size:
            chunk_text = tokenizer.decode(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            current_chunk = []
            current_length = 0
    
    if current_chunk:
        chunk_text = tokenizer.decode(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
    
    return chunks

class StylometryPipeline:
    def __init__(self, input_dir: str = "data/input", 
                 processed_dir: str = "data/processed",
                 error_dir: str = "data/error",
                 results_dir: str = "data/embedding_results"):
        self.logger = setup_logging()
        
        # Set up directories
        self.input_dir = Path(input_dir)
        self.processed_dir = Path(processed_dir)
        self.error_dir = Path(error_dir)
        self.results_dir = Path(results_dir)
        
        for dir_path in [self.input_dir, self.processed_dir, self.error_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ML components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        self.logger.info("Loading NLP models...")
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        self.model = AutoModel.from_pretrained('AIDA-UPM/star').to(self.device)
        self.model.eval()

    def process_chunk(self, chunk: str) -> np.ndarray:
        """Process a single chunk of text and return its embedding."""
        try:
            inputs = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.pooler_output.cpu().numpy()

            return embedding[0]
        except Exception as e:
            self.logger.error(f"Error processing chunk: {str(e)}")
            raise

    def process_file(self, filepath: Path) -> Tuple[str, np.ndarray]:
        """Process a single text file with chunking."""
        self.logger.info(f"Processing file: {filepath}")
        
        try:
            # Read file
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            # Chunk text
            chunks = chunk_text(text, self.tokenizer)
            self.logger.info(f"Split into {len(chunks)} chunks")

            # Process each chunk
            chunk_embeddings = []
            for chunk in tqdm(chunks, desc="Processing chunks"):
                embedding = self.process_chunk(chunk)
                chunk_embeddings.append(embedding)

            job_id = f"job_{filepath.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create directory for chunk embeddings
            chunk_dir = self.results_dir / job_id / 'chunks'
            chunk_dir.mkdir(parents=True, exist_ok=True)
            
            # Process chunks and save individually
            chunk_embeddings = []
            for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
                embedding = self.process_chunk(chunk)
                chunk_embeddings.append(embedding)
                
                # Save individual chunk embedding
                chunk_result = {
                    'chunk_text': chunk,
                    'embedding': embedding.tolist(),
                    'chunk_index': i
                }
                with open(chunk_dir / f'chunk_{i}.json', 'w') as f:
                    json.dump(chunk_result, f, indent=2)

            # Average embeddings
            if not chunk_embeddings:
                raise ValueError("No valid chunks processed")
                
            final_embedding = np.mean(chunk_embeddings, axis=0)
            
            # Save results
            result = {
                job_id: {
                    'text': text,  # original text
                    'embedding': final_embedding.tolist(),
                    'metadata': {
                        'filename': filepath.name,
                        'processed_at': datetime.now().isoformat(),
                        'num_chunks': len(chunks),
                        'file_size': filepath.stat().st_size
                    }
                }
            }
            
            # Save to JSON
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = self.results_dir / f'embedding_{filepath.stem}_{timestamp}.json'
            
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Move processed file
            dest_path = self.processed_dir / filepath.name
            shutil.move(str(filepath), str(dest_path))
            
            self.logger.info(f"Successfully processed {filepath}")
            return job_id, final_embedding

        except Exception as e:
            self.logger.error(f"Error processing file {filepath}: {str(e)}")
            error_path = self.error_dir / filepath.name
            shutil.move(str(filepath), str(error_path))
            raise

def main():
    pipeline = StylometryPipeline()
    
    # Process all txt files in input directory
    for filepath in pipeline.input_dir.glob("*.txt"):
        try:
            pipeline.process_file(filepath)
        except Exception as e:
            pipeline.logger.error(f"Failed to process {filepath}: {str(e)}")
            continue

if __name__ == "__main__":
    main()