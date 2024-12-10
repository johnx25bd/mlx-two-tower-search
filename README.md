# MLX5 Week 2: Two Tower Search

A semantic search engine implementation using a two-tower neural network architecture for efficient similarity search and retrieval. This project demonstrates modern machine learning techniques for information retrieval and scalable search solutions.

![architecture](public/images/architecture.png)

## Project Overview

This two-tower search model implements a dual-encoder (two-tower) neural network architecture for semantic search, allowing for:
- Fast similarity search across large document collections
- Efficient vector representations of both queries and documents
- Real-time search capabilities with pre-computed document embeddings
- Scalable architecture suitable for production deployments

Built with [@jigisha-p](https://github.com/jigisha-p) and [@kalebsofer](https://github.com/kalebsofer) 

## Note

This codebase won't work out of the box — it relies on a few files that are difficult to include in this repo due to file size constraints. Specifically, it relies on a .pth file for the model weights, a .faiss index file, and a .parquet file for the document data.

While this doesn't compile, we do have a live deployment running at [simplesearchengine.com](https://simplesearchengine.com). (We haven't configured HTTPS, so you may need to click through the warning if want to see the deployed site.)

The mlx-deploy-search repository contains the code for the live deployment.

## Key Features

- **Two-Tower Neural Architecture**: Separate encoding paths for queries and documents, optimizing for both accuracy and inference speed
- **Vector Similarity Search**: Efficient nearest neighbor search using cosine similarity
- **FastAPI Backend**: Modern, high-performance API server with automatic OpenAPI documentation
- **Streamlit Frontend**: Interactive web interface for real-time search demonstrations
- **Docker Support**: Containerized deployment for easy scaling and reproducibility

## Technical Architecture

The system consists of three main components:

1. **Query Tower**: Processes and encodes search queries into dense vector representations
2. **Document Tower**: Transforms documents into semantic vectors for efficient matching
3. **Search Infrastructure**: 
   - Vector similarity computation
   - Efficient index structures for fast retrieval
   - API endpoints for search operations

## Use Cases

This implementation is particularly suitable for:
- Content recommendation systems
- Document retrieval systems
- Semantic search engines
- Similar item finding
- Knowledge base search

## Performance

The two-tower architecture offers several advantages:
- Document vectors can be pre-computed and cached
- Real-time query processing with minimal latency
- Scalable to millions of documents
- Efficient memory usage through dimensional reduction


## Local Setup

Create virtual environment:

```bash
python -m venv env
source env/bin/activate
```

Install dependencies (python 3.12):

```bash
pip install -r requirements.txt
``` 

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

In a seperate terminal, start the streamlit app:

```bash
streamlit run streamlit_app.py
```

## Local Docker Deployment

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed on your machine.

## Steps

Navigate to project directory and build Docker Image
   ```bash
   docker build -t twotowersearch .
   ```

Run Docker Container
   ```bash
   docker run -p 8000:8000 twotowersearch
   ```

Open a web browser and navigate to `http://localhost:8000` to access the FastAPI server.

Stop Docker Container

```bash
docker stop <container_id>
```



## API Documentation

### Endpoints

- `POST /search`: Performs semantic search
  ```json
  {
    "query": "search query",
    "top_k": 5
  }
  ```

- `GET /health`: System health check

### Response Format

```json
{
    "results": [
        {
            "document_id": "doc1",
            "similarity_score": 0.89,
            "content": "..."
        }
    ]
}
```

## Development

### Project Structure
```
twotowersearch/
├── app/
│   ├── models/
│   ├── api/
│   └── core/
├── tests/
├── docker/
└── streamlit_app.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with PyTorch and Transformers
- Inspired by Google's dual encoder architecture
- Vector similarity powered by FAISS

