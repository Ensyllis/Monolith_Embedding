import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import numpy as np
import umap.umap_ as umap
import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class MemoryManager:
    def __init__(self, max_items: int = 1000, cleanup_threshold: float = 0.9):
        self.max_items = max_items
        self.cleanup_threshold = cleanup_threshold

    def check_and_cleanup(self, results: Dict) -> Dict:
        if len(results) > self.max_items * self.cleanup_threshold:
            # Sort by timestamp and keep only the most recent max_items
            sorted_items = sorted(
                results.items(),
                key=lambda x: x[1]['metadata']['processed_at'],
                reverse=True
            )
            return dict(sorted_items[:self.max_items])
        return results

def setup_logging(log_dir: str = "logs", max_bytes: int = 10485760, backup_count: int = 5) -> logging.Logger:
    """Set up logging with rotation and separate files for different log levels."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("StylometryPipeline")
    logger.setLevel(logging.DEBUG)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create handlers for different log levels
    handlers = {
        'debug': RotatingFileHandler(
            log_dir / 'debug.log', maxBytes=max_bytes, backupCount=backup_count
        ),
        'info': RotatingFileHandler(
            log_dir / 'info.log', maxBytes=max_bytes, backupCount=backup_count
        ),
        'error': RotatingFileHandler(
            log_dir / 'error.log', maxBytes=max_bytes, backupCount=backup_count
        )
    }

    # Set levels and formatters
    handlers['debug'].setLevel(logging.DEBUG)
    handlers['debug'].setFormatter(detailed_formatter)
    handlers['info'].setLevel(logging.INFO)
    handlers['info'].setFormatter(simple_formatter)
    handlers['error'].setLevel(logging.ERROR)
    handlers['error'].setFormatter(detailed_formatter)

    # Add handlers to logger
    for handler in handlers.values():
        logger.addHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    return logger

class StylometryPipeline:
    def __init__(
            self, 
            input_dir: str = "data/input", 
            processed_dir: str = "data/processed", 
            results_dir: str = "data/embedding_results", 
            max_memory_items: int = 100
        ):

        # Set up logging
        self.logger = setup_logging()
        
        # Validate directories and permissions
        self.validate_directories(input_dir, processed_dir, results_dir)
        
        # Set up memory management
        self.memory_manager = MemoryManager(max_items=max_memory_items)
        
        # Initialize ML components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.logger.info("Loading NLP models...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')
            self.model = AutoModel.from_pretrained('AIDA-UPM/star').to(self.device)
            self.model.eval()
        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            raise

        self.umap_reducer = umap.UMAP(
            n_neighbors=15,
            n_components=2,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )

        # Store results in memory and load any existing results
        self.results = self.load_results()

    def validate_directories(
            self, 
            input_dir: str, 
            processed_dir: str, 
            results_dir: str
        ):
        """Validate and create directories with proper permissions."""
        try:
            for dir_path in [input_dir, processed_dir, results_dir]:
                path = Path(dir_path)
                path.mkdir(parents=True, exist_ok=True)
                
                # Test write permissions
                test_file = path / '.permission_test'
                try:
                    test_file.touch()
                    test_file.unlink()
                except Exception as e:
                    self.logger.error(f"No write permission in {dir_path}: {str(e)}")
                    raise PermissionError(f"No write permission in {dir_path}")
                
            self.input_dir = Path(input_dir)
            self.processed_dir = Path(processed_dir)
            self.results_dir = Path(results_dir)
            
            self.logger.info("Directory validation successful")
        except Exception as e:
            self.logger.error(f"Directory validation failed: {str(e)}")
            raise

    def cleanup_old_results(
            self, 
            max_files: int = 10
        ):
        """Clean up old result files keeping only the most recent ones."""
        result_files = list(self.results_dir.glob('embeddings_*.json'))
        if len(result_files) > max_files:
            sorted_files = sorted(result_files, key=lambda x: x.stat().st_mtime)
            for file in sorted_files[:-max_files]:
                try:
                    file.unlink()
                    self.logger.info(f"Cleaned up old result file: {file}")
                except Exception as e:
                    self.logger.error(f"Failed to delete old result file {file}: {str(e)}")

    def save_results(self):
        """Save results to JSON file with cleanup."""
        try:
            # Clean up old result files
            self.cleanup_old_results()
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for job_id, data in self.results.items():
                serializable_results[job_id] = {
                    'text': data['text'],
                    'embedding': data['embedding'].tolist(),
                    'metadata': data['metadata'],
                    'coordinates': data.get('coordinates', None)
                }

            # Save to a timestamped JSON file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = self.results_dir / f'embeddings_{timestamp}.json'
            
            with open(result_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"Results saved to {result_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise

    def process_file(
            self, 
            filepath: Path
        ) -> Tuple[str, np.ndarray]:
        """Process a single text file with enhanced error handling."""
        self.logger.info(f"Processing file: {filepath}")
        
        try:
            # Check file size before processing
            file_size = filepath.stat().st_size
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError(f"File too large: {file_size / (1024*1024):.2f}MB")

            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            embedding = self.process_text(text)
            job_id = f"job_{filepath.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.results[job_id] = {
                'text': text,
                'embedding': embedding,
                'metadata': {
                    'filename': filepath.name,
                    'processed_at': datetime.now().isoformat(),
                    'file_size': file_size
                }
            }

            # Memory management
            self.results = self.memory_manager.check_and_cleanup(self.results)

            # Save results and move file
            self.save_results()
            dest_path = self.processed_dir / filepath.name
            shutil.move(str(filepath), str(dest_path))
            self.logger.info(f"Successfully processed and moved file: {filepath}")

            return job_id, embedding

        except Exception as e:
            self.logger.error(f"Error processing file {filepath}: {str(e)}")
            # Move failed files to an error directory
            error_dir = self.input_dir / "error"
            error_dir.mkdir(exist_ok=True)
            try:
                shutil.move(str(filepath), str(error_dir / filepath.name))
                self.logger.info(f"Moved failed file to error directory: {filepath}")
            except Exception as move_error:
                self.logger.error(f"Failed to move error file: {str(move_error)}")
            raise

class TextFileHandler(FileSystemEventHandler):
    def __init__(self, pipeline: StylometryPipeline):
        self.pipeline = pipeline
        self.logger = logging.getLogger("StylometryPipeline.FileHandler")

    def on_created(self, event):
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        if filepath.suffix.lower() == '.txt':
            try:
                job_id, _ = self.pipeline.process_file(filepath)
                self.logger.info(f"Processed new file. Job ID: {job_id}")
                self.pipeline.analyze_embeddings()
            except Exception as e:
                self.logger.error(f"Error handling new file: {str(e)}")

def main():
    try:
        pipeline = StylometryPipeline()
        pipeline.logger.info("Pipeline initialized")

        # Process existing files
        for filepath in pipeline.input_dir.glob("*.txt"):
            try:
                pipeline.process_file(filepath)
            except Exception as e:
                pipeline.logger.error(f"Error processing existing file {filepath}: {str(e)}")

        # Analyze initial embeddings
        if pipeline.results:
            pipeline.analyze_embeddings()

        # Watch for new files
        handler = TextFileHandler(pipeline)
        observer = Observer()
        observer.schedule(handler, str(pipeline.input_dir), recursive=False)
        observer.start()
        pipeline.logger.info(f"Watching for new files in {pipeline.input_dir}")

        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        pipeline.logger.info("Stopping file observer")
    except Exception as e:
        pipeline.logger.error(f"Critical error in main loop: {str(e)}")
        raise
    finally:
        observer.join()
        pipeline.logger.info("Pipeline shutdown complete")

if __name__ == "__main__":
    main()