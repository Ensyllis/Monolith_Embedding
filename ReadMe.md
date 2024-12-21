# Writing Style Analysis System: From Text to Embeddings

## Deploying the code

### Setting Up Your Environment

First, we need to create a home for our project. Open your terminal and run:

```bash
# Get the code from GitHub 
git clone https://github.com/Ensyllis/Monolith_Embedding.git 
cd Monolith_Embedding
```

### Data Input:

Put all of your .text files inside of the data/input folder

### Docker Deployment:

The system uses Docker to manage dependencies and ensure consistent execution across environments. Deploy using:
```bash
docker-compose build
docker-compose up -d

# Monitor system execution
docker-compose logs -f
```

This will automatically start running the system for embedding process in the data/input folder.
### System Monitoring:

Track proccessing status and usage through the logging system

```bash
# View processing logs
tail -f data/logs/pipeline.log

# Check processed files and results
ls -l data/processed/
ls -l data/embedding_results/
```
### Analysis Pipeline

To analyze writing styles, place text files in the input directory and initiate processing:

```bash
# Initiate the analysis pipeline 
curl -X POST http://localhost:5000/generate_umap
```

This will call the API backend which will initiate the analysis section of the code. 

### Error Handling

The system implements a robust error handling mechanism. Failed files are automatically moved to the error directory for investigation and reprocessing:

```bash
# Review failed files
ls -l data/error/

# Reprocess failed files
mv data/error/* data/input/
```

---
---
---
## High-Level Overview

This system transforms written text into mathematical representations (embeddings) that capture writing style. These embeddings allow us to:
- Compare writing styles numerically
- Group similar writing styles
- Detect stylistic outliers
- Attribute potential authorship

### Core Concept Example

Consider these messages from different authors:

```
Author 1: "Hello there! How are you doing today?"
Author 2: "sup bro"
Author 3: "Greetings, I trust this message finds you well."
```

These differences in formality, punctuation, and word choice create distinct "fingerprints" that our system captures mathematically.

## System Architecture

1. **Input Processing**: Text files → Chunks
2. **Embedding Generation**: Chunks → Numerical Vectors
3. **Result Storage**: Vectors → Structured JSON
4. **Error Handling**: Automated recovery and logging

## Technical Deep Dive

Let's follow a sample academic text through the system:

```text
The scribe's particular choice of vocabulary indicates a formal education in ecclesiastical matters. 
The intersection of paleographic evidence and textual analysis yields surprising correlations. 
Upon careful examination of the extant manuscripts, several distinct patterns emerge...
```

### Step 1: Pipeline Initialization
```python
def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = AutoModel.from_pretrained('AIDA-UPM/star').to(self.device)
    self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')
```

The system loads:
- AIDA-UPM/star model (fine-tuned RoBERTa)
- GPU acceleration if available
- RoBERTa tokenizer for text processing

### Step 2: Text Chunking

Input text gets split into 512-token chunks (model's maximum input size):

```python
chunks = chunk_text(text, self.tokenizer)
```

Example chunks:
```python
[
    "The scribe's particular choice of vocabulary indicates... correlations.",
    "Upon careful examination of the extant manuscripts... patterns emerge."
]
```

Each chunk maintains semantic coherence while fitting within model constraints.

### Step 3: Embedding Generation

Each chunk becomes a 768-dimensional vector:

```python
def process_chunk(self, chunk: str) -> np.ndarray:
    inputs = self.tokenizer(chunk, padding=True, truncation=True)
    outputs = self.model(**inputs)
    return outputs.pooler_output.cpu().numpy()
```

Chunk → Tokens → Model → 768D Vector:
```python
[0.13589645, 0.02458921, ..., 0.05678901]
```

### Step 4: Result Storage

Results get saved in a structured hierarchy:

```
results_dir/
├── job_academic_20241221_102030/
│   └── chunks/
│       ├── chunk_0.json  # Individual chunk data
│       └── chunk_1.json
└── embedding_academic_20241221_102030.json  # Final averaged embedding
```

Each chunk file contains:
```json
{
    "chunk_text": "Original text segment",
    "embedding": [vector values],
    "chunk_index": 0
}
```

Final embedding file contains:
```json
{
    "job_academic_20241221_102030": {
        "text": "Complete original text",
        "embedding": [averaged vector],
        "metadata": {
            "filename": "academic.txt",
            "processed_at": "2024-12-21T10:20:30",
            "num_chunks": 2,
            "file_size": 1024
        }
    }
}
```

## System Limitations

1. **Memory Constraints**
   - Maximum 512 tokens per chunk
   - GPU memory usage scales with batch size

2. **Processing Limits**
   - Processes one file at a time
   - Large files require chunking overhead

3. **Edge Cases**
   - May split mid-sentence at chunk boundaries
   - Very short texts might lose stylistic nuance

## Usage

1. Place .txt files in `data/input/`
2. Run:
   ```bash
   docker-compose build up
   ```
3. Monitor `data/logs/` for progress
4. Find results in `data/embedding_results/`
5. Access analysis via:
   ```
   POST http://localhost:5000/generate_umap
   ```

## Error Recovery

- Failed files move to `data/error/`
- Successfully processed files move to `data/processed/`
- System can resume interrupted processing
- Detailed logs track all operations

The system is designed to be robust, handling everything from Shakespeare to tweets with equal aplomb. Just feed it text, and it'll give you numbers that capture the essence of the writing style.


---
---
---

# Section 2: Visualization and Analysis

## Overview
Once we've generated embeddings, we need to make sense of these 1024-dimensional vectors. The system provides a Flask-based visualization service that reduces these complex vectors into a 2D representation using UMAP (Uniform Manifold Approximation and Projection). Note: You can readjust this portion of the text however you wish. 

## Technical Components

### Data Loading
```python
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
```
- Scans `data/embedding_results` directory
- Extracts embeddings and their IDs
- Returns arrays ready for visualization

### Dimensionality Reduction
```python
reducer = umap.UMAP(
    n_neighbors=15,
    n_components=2,
    min_dist=0.1, 
    metric='euclidean'
)
embedding_2d = reducer.fit_transform(embeddings)
```

UMAP Configuration:
- `n_neighbors=15`: Balance between local and global structure
- `min_dist=0.1`: Compactness of visualization
- `metric='euclidean'`: Standard distance measurement

### Visualization Output
```
data/visualizations/
└── embedding_visualization.png
```

The system generates:
- 2D scatter plot
- Point labels showing job IDs
- Color gradient for visual grouping
- High-resolution output (300 DPI)

## API Endpoint

```bash
POST http://localhost:5000/generate_umap
```

### Response Format
```json
{
    "status": "success",
    "message": "UMAP visualization generated",
    "coords": [[x1, y1], [x2, y2], ...],
    "ids": ["job_1", "job_2", ...]
}
```

### Error Response
```json
{
    "status": "error",
    "message": "Error details"
}
```

## Interpreting Results

The visualization reveals:
- Clustered points: Similar writing styles
- Distant points: Distinct styles
- Gradients: Style transitions

Example:
```
Academic texts -----> Blog posts -----> Social media
[Formal cluster]    [Mixed styles]    [Informal cluster]
```

## Usage Tips

1. More samples = better visualization
2. Similar styles cluster naturally
3. Outliers often indicate unique writing patterns
4. Distance between points roughly correlates to style similarity

## Directory Structure After Analysis
```
data/
├── embedding_results/    # Input embeddings
├── visualizations/       # Output plots
└── logs/                # Processing logs
```

## Performance Notes
- UMAP processing time scales with sample size
- Large datasets (>1000 samples) may take several minutes
- GPU acceleration not implemented for visualization

## Common Pitfalls
1. Insufficient samples for meaningful clusters
2. Overinterpreting small distances
3. Not accounting for text length variations
4. Missing genre/context in interpretation

Remember: The visualization is a tool for exploration, not a definitive style categorization. Use it to identify patterns and generate hypotheses about writing style relationships.