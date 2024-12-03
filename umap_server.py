from flask import Flask, jsonify
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import umap
import logging
from typing import Dict, List, Tuple

app = Flask(__name__)

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def load_embeddings(results_dir: str) -> Tuple[np.ndarray, List[str]]:
    embeddings = []
    ids = []
    
    for f in Path(results_dir).glob('*.json'):
        with open(f, 'r') as file:
            data = json.load(file)
            for job_id, content in data.items():
                embeddings.append(content['embedding'])
                ids.append(job_id)
    
    return np.array(embeddings), ids

def create_visualization(embeddings: np.ndarray, ids: List[str], output_dir: str):
    # UMAP reduction
    reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='euclidean')
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=range(len(ids)), cmap='viridis')
    
    # Add labels for each point
    for i, txt in enumerate(ids):
        plt.annotate(txt, (embedding_2d[i, 0], embedding_2d[i, 1]), fontsize=8, alpha=0.7)
    
    plt.colorbar(scatter)
    plt.title('UMAP Visualization of Document Embeddings')
    
    # Save plot
    output_path = Path(output_dir) / 'embedding_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return embedding_2d

@app.route('/generate_umap', methods=['POST'])
def generate_umap():
    logger = setup_logging()
    try:
        results_dir = "data/embedding_results"
        output_dir = "data/visualizations"
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load embeddings
        embeddings, ids = load_embeddings(results_dir)
        logger.info(f"Loaded {len(embeddings)} embeddings")
        
        # Generate visualization
        coords = create_visualization(embeddings, ids, output_dir)
        logger.info("Generated UMAP visualization")
        
        return jsonify({
            "status": "success",
            "message": "UMAP visualization generated",
            "coords": coords.tolist(),
            "ids": ids
        })
        
    except Exception as e:
        logger.error(f"Error generating UMAP: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)