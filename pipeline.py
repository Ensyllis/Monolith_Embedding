import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import numpy as np
import umap.umap_ as umap
import logging
from typing import Dict, List, Tuple
import json
from datetime import datetime
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StylometryPipeline:
    def __init__(self, input_dir: str = "data/input", processed_dir: str = "data/processed", 
                 results_dir: str = "data/embedding_results"):
        # Set up directories
        self.input_dir = Path(input_dir)
        self.processed_dir = Path(processed_dir)
        self.results_dir = Path(results_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ML components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        logger.info("Loading NLP models...")
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        self.model = AutoModel.from_pretrained('AIDA-UPM/star').to(self.device)
        self.model.eval()

        self.umap_reducer = umap.UMAP(
            n_neighbors=15,
            n_components=2,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )

        # Store results in memory and load any existing results
        self.results = self.load_results()

    def save_results(self):
        """Save results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for job_id, data in self.results.items():
            serializable_results[job_id] = {
                'text': data['text'],
                'embedding': data['embedding'].tolist(),  # Convert numpy array to list
                'metadata': data['metadata'],
                'coordinates': data.get('coordinates', None)  # Include UMAP coordinates if they exist
            }

        # Save to a timestamped JSON file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = self.results_dir / f'embeddings_{timestamp}.json'
        
        with open(result_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {result_file}")

    def load_results(self) -> Dict:
        """Load most recent results if they exist."""
        result_files = list(self.results_dir.glob('embeddings_*.json'))
        if not result_files:
            return {}

        # Get most recent results file
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            for job_id in data:
                data[job_id]['embedding'] = np.array(data[job_id]['embedding'])
            
            logger.info(f"Loaded previous results from {latest_file}")
            return data
        except Exception as e:
            logger.error(f"Error loading previous results: {e}")
            return {}

    def process_text(self, text: str) -> np.ndarray:
        """Convert text into embedding."""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.pooler_output.cpu().numpy()

        return embedding[0]

    def process_file(self, filepath: Path) -> Tuple[str, np.ndarray]:
        """Process a single text file and generate its embedding."""
        logger.info(f"Processing file: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            embedding = self.process_text(text)
            job_id = f"job_{filepath.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.results[job_id] = {
                'text': text,
                'embedding': embedding,
                'metadata': {
                    'filename': filepath.name,
                    'processed_at': datetime.now().isoformat()
                }
            }

            # Save results after each file is processed
            self.save_results()

            # Move file to processed directory
            dest_path = self.processed_dir / filepath.name
            shutil.move(str(filepath), str(dest_path))

            return job_id, embedding

        except Exception as e:
            logger.error(f"Error processing file {filepath}: {str(e)}")
            raise

    def analyze_embeddings(self, job_ids: List[str] = None) -> Dict:
        """Analyze embeddings using UMAP."""
        if job_ids is None:
            job_ids = list(self.results.keys())

        if not job_ids:
            logger.warning("No embeddings to analyze")
            return {}

        embeddings = np.array([self.results[job_id]['embedding'] for job_id in job_ids])
        coords = self.umap_reducer.fit_transform(embeddings)

        # Store coordinates with results
        for idx, job_id in enumerate(job_ids):
            self.results[job_id]['coordinates'] = coords[idx].tolist()

        # Save updated results with coordinates
        self.save_results()

        return {
            job_id: {
                'coordinates': self.results[job_id]['coordinates'],
                'metadata': self.results[job_id]['metadata']
            }
            for job_id in job_ids
        }

class TextFileHandler(FileSystemEventHandler):
    def __init__(self, pipeline: StylometryPipeline):
        self.pipeline = pipeline

    def on_created(self, event):
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        if filepath.suffix.lower() == '.txt':
            try:
                job_id, _ = self.pipeline.process_file(filepath)
                logger.info(f"Processed new file. Job ID: {job_id}")
                self.pipeline.analyze_embeddings()
            except Exception as e:
                logger.error(f"Error handling new file: {str(e)}")

def main():
    pipeline = StylometryPipeline()
    logger.info("Pipeline initialized")

    # Process existing files
    for filepath in pipeline.input_dir.glob("*.txt"):
        try:
            pipeline.process_file(filepath)
        except Exception as e:
            logger.error(f"Error processing existing file {filepath}: {str(e)}")

    # Analyze initial embeddings
    if pipeline.results:
        pipeline.analyze_embeddings()

    # Watch for new files
    handler = TextFileHandler(pipeline)
    observer = Observer()
    observer.schedule(handler, str(pipeline.input_dir), recursive=False)
    observer.start()
    logger.info(f"Watching for new files in {pipeline.input_dir}")

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Stopping file observer")
    observer.join()

if __name__ == "__main__":
    main()