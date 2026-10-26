🚀 VectorForge — A Vector Database Built from Scratch in Python

<p align="center">
<strong>Explore Vector Search, HNSW, Semantic Retrieval, and Local RAG.</strong>
</p>

<p align="center">
An educational vector database that brings together multiple nearest-neighbor search algorithms, interactive vector visualization, real document embeddings, and AI-powered question answering.
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/USearch-HNSW-6C5CE7?style=for-the-badge" alt="USearch HNSW">
<img src="https://img.shields.io/badge/Ollama-Local_AI-black?style=for-the-badge" alt="Ollama">
<img src="https://img.shields.io/badge/RAG-Enabled-00A67E?style=for-the-badge" alt="RAG">
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License">
</p>

📌 Overview

VectorForge is a Python-based vector database project designed to demonstrate how modern vector search systems work internally.

It implements and compares three nearest-neighbor search approaches:

HNSW — fast approximate nearest-neighbor search powered by USearch.
KD-Tree — spatial partitioning for nearest-neighbor search.
Brute Force — exact distance calculations used as a baseline.

The project also explores semantic embeddings, dimensionality reduction, document retrieval, and Retrieval-Augmented Generation (RAG) using locally running AI models through Ollama.

Rather than treating a vector database as a black box, VectorForge makes the underlying concepts accessible through a unified search interface and a web-based experience.

💡 The goal: Understand how vectors are stored, indexed, searched, visualized, and used to give language models relevant context.
✨ Features
Feature	Description
🔎 Multiple Search Algorithms	Compare HNSW, KD-Tree, and Brute Force
⚡ USearch Integration	Efficient HNSW-based approximate nearest-neighbor indexing
📐 Distance Metrics	Cosine, Euclidean, and Manhattan distance, where supported
🧠 Semantic Search	Find conceptually similar vectors and documents
📊 Vector Visualization	Explore high-dimensional vectors in a 2D PCA projection
📄 Document Ingestion	Embed text and index document chunks
✂️ Text Chunking	Split longer documents into manageable, overlapping chunks
🤖 Local AI	Run embedding and generation models through Ollama
💬 RAG Pipeline	Retrieve relevant passages and generate context-aware answers
🌐 Web Interface	Explore searches, results, visualizations, and AI responses
🔌 REST API	Access supported search, indexing, document, and RAG operations
🔒 Local-First Workflow	Keep inference local when all configured services run locally
🖼️ Application Preview
Add your own screenshots here to showcase the actual application. GitHub will render the images once you commit them to your repository.

Recommended screenshots:

Search Interface	Vector Visualization	RAG Chat
docs/images/search.png	docs/images/vectors.png	docs/images/rag.png

Once you have added the screenshots, replace the table above with:

markdown
<p align="center">
  <img src="docs/images/search.png" width="90%" alt="VectorForge search interface">
</p>

<p align="center">
  <img src="docs/images/vectors.png" width="90%" alt="Vector space visualization">
</p>

<p align="center">
  <img src="docs/images/rag.png" width="90%" alt="RAG question answering interface">
</p>

🏗️ System Architecture

VectorForge has two primary workflows: nearest-neighbor search and document question answering.

1. Vector Search Engine
text
                  Query Vector
                       |
                       v
              +------------------+
              |   VectorForge    |
              |   Search Engine  |
              +------------------+
                       |
          +------------+------------+
          |            |            |
          v            v            v
     Brute Force     KD-Tree    HNSW / USearch
          |            |            |
          +------------+------------+
                       |
                       v
               Top-K Neighbors
                       |
                       v
                Search Results

2. Retrieval-Augmented Generation (RAG)
text
                 Your Documents
                       |
                       v
                 Text Chunking
                       |
                       v
                Embedding Model
               nomic-embed-text
                       |
                       v
                Vector Index
                HNSW / USearch
                       |
                       |
                  User Question
                       |
                       v
               Embed the Question
                       |
                       v
                Semantic Search
                       |
                       v
                Relevant Chunks
                       |
                       v
                 Local LLM
                  llama3.2
                       |
                       v
                  AI Answer


The embedding model converts text into numerical vectors. The vector index retrieves relevant content, and the language model uses the retrieved context to formulate an answer.

🔍 Search Algorithms

One of the main goals of VectorForge is to demonstrate how different nearest-neighbor algorithms behave on the same dataset.

⚡ HNSW — Hierarchical Navigable Small World

HNSW is a graph-based approximate nearest-neighbor algorithm widely used in vector search systems.

VectorForge uses USearch for HNSW indexing.

How it works:

Each vector is represented by a node in a proximity graph.
Nodes can participate in multiple layers.
Upper layers provide long-range navigation through the graph.
Lower layers refine the search among nearby candidates.
A query traverses the graph to find its nearest neighbors.

Advantages:

Fast approximate search for many high-dimensional workloads.
Useful for semantic search and embedding retrieval.
Offers configurable index parameters and supported distance metrics.

Trade-offs:

May not return the exact nearest neighbors.
Requires additional index memory.
Performance depends on index configuration and dataset characteristics.
🌳 KD-Tree

A KD-Tree recursively partitions a vector space along selected dimensions.

How it works:

Choose a dimension and split the data around a threshold.
Recursively partition the resulting regions.
Search promising branches first.
Prune branches that cannot improve the current nearest-neighbor results.

Advantages:

Can be effective for low-dimensional data.
Supports exact nearest-neighbor search when implemented with correct pruning.
Provides an excellent way to learn spatial indexing.

Trade-offs:

Performance often degrades as dimensionality increases.
Search can approach a linear scan in difficult workloads.
🔬 Brute Force

Brute Force compares a query vector against every stored vector.

Advantages:

Simple and predictable.
Provides exact nearest-neighbor results for the chosen distance metric.
Useful as ground truth when evaluating approximate search.

Trade-offs:

Query cost grows with dataset size.
Repeated distance calculations become expensive for large datasets.
Algorithm Comparison
Property	Brute Force	KD-Tree	HNSW
Search type	Exact	Exact with correct pruning	Approximate
Low-dimensional data	Good baseline	Often effective	Often effective
High-dimensional data	Predictable linear scan	Pruning may degrade	Often effective
Index overhead	Low	Additional tree structure	Additional graph structure
Best role	Ground truth	Spatial search	Fast approximate retrieval

For 
𝑁
 vectors with 
𝑑
 dimensions, Brute Force typically requires 
𝑂
(
𝑁
𝑑
)
 work per query. KD-Tree performance depends on dimensionality and pruning effectiveness. HNSW often achieves much faster practical search, but its theoretical and actual performance depends on graph parameters and workload.

📐 Distance Metrics

VectorForge can demonstrate different distance functions, subject to support in each search backend.

Cosine Similarity

Measures the directional similarity between two vectors.

cosine
⁡
(
𝐴
,
𝐵
)
=
𝐴
⋅
𝐵
∥
𝐴
∥
∥
𝐵
∥

Higher similarity indicates more similar directions. If the implementation returns cosine distance instead, lower values indicate closer vectors.

Euclidean Distance

Measures the straight-line distance between vectors.

𝑑
(
𝐴
,
𝐵
)
=
∑
𝑖
=
1
𝑑
(
𝐴
𝑖
−
𝐵
𝑖
)
2

Lower values indicate closer vectors.

Manhattan Distance

Measures the sum of absolute differences between dimensions.

𝑑
(
𝐴
,
𝐵
)
=
∑
𝑖
=
1
𝑑
∣
𝐴
𝑖
−
𝐵
𝑖
∣

Lower values indicate closer vectors.

Implementation note: The selected metric must be consistent with index configuration, query calculations, and result ranking. Confirm that each algorithm supports the selected metric before comparing results.

📊 Visualizing Vector Space with PCA

High-dimensional embeddings are difficult to interpret directly. Principal Component Analysis (PCA) projects vectors into a lower-dimensional space for visualization.

A 2D scatter plot can help illustrate:

How related concepts cluster together.
How different categories are distributed.
Where a query lies relative to stored vectors.
How retrieved neighbors relate to the query.

PCA is a visualization technique, not a replacement for the original embeddings. Two points that appear close in 2D may not be the nearest neighbors in the original high-dimensional space.

🤖 Local AI with Ollama

VectorForge integrates with Ollama to run embedding and text-generation models locally.

Model	Purpose
nomic-embed-text	Converts text into vector embeddings
llama3.2	Generates answers using retrieved document context

The standard nomic-embed-text model typically produces 768-dimensional embeddings.

The embedding and generation models perform different tasks: one represents text numerically, while the other generates natural-language responses.

Local inference avoids a hosted AI API when the application is configured to use only local services. Verify the application's network behavior before relying on it for sensitive data.

📚 Retrieval-Augmented Generation (RAG)

RAG combines semantic retrieval with language generation. Instead of relying exclusively on a model's pretrained knowledge, the application retrieves relevant passages from your documents and supplies them as context.

How the Pipeline Works
Ingest: Provide text or a document.
Chunk: Split long documents into smaller, overlapping passages.
Embed: Generate a vector for each chunk.
Index: Store the embeddings and associated text.
Retrieve: Embed the user's question and search for relevant chunks.
Generate: Pass the question and retrieved context to the local LLM.
Answer: Return the generated response and, where supported, show the retrieved passages.
Example

Suppose you insert notes about operating systems and ask:

What is the difference between a process and a thread?

The system embeds your question, retrieves relevant passages from the indexed notes, and supplies them to the language model as context.

The resulting answer depends on the quality of the source material, chunking strategy, embeddings, retrieval configuration, and language model.

⚠️ Important: RAG does not guarantee factual correctness. A language model may still produce unsupported statements, even when relevant context is provided.
⚙️ Prerequisites

Before running VectorForge, install:

Python 3.10 or later — adjust to your project's actual supported version.
Git — for cloning the repository.
Ollama — for local embeddings and generation.
Project dependencies — as specified by your requirements.txt or pyproject.toml.

USearch supports multiple platforms. Refer to its official repository for platform-specific installation guidance.

🚀 Installation and Setup
Step 1 — Clone the Repository
bash
git clone https://github.com/YOUR_USERNAME/vector-forge.git
cd vector-forge


Replace YOUR_USERNAME with your GitHub username.

Step 2 — Create a Virtual Environment

Windows PowerShell

powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1


Linux / macOS

bash
python3 -m venv .venv
source .venv/bin/activate

Step 3 — Install Dependencies

If your repository contains requirements.txt:

bash
python -m pip install --upgrade pip
pip install -r requirements.txt


If you use pyproject.toml or another dependency manager, follow the installation instructions for that setup.

Step 4 — Install Ollama Models

Download and install Ollama from ollama.com.

Pull the embedding model:

bash
ollama pull nomic-embed-text


Pull the generation model:

bash
ollama pull llama3.2


Verify the installed models:

bash
ollama list


Ollama generally runs as a background service after installation. If the service is not running, start it:

bash
ollama serve


Do not start a second server if Ollama is already running.

Step 5 — Start VectorForge

Use the entry point appropriate for your repository. For example, if your application starts in main.py:

bash
python main.py


Open the local address printed by your application in a browser.

Note: The actual entry-point filename, web framework, and port depend on your implementation. Replace this example with the exact command users should run.

🖥️ Using the Application
Vector Search
Enter a query or select a sample vector.
Choose HNSW, KD-Tree, or Brute Force.
Select a supported distance metric.
Choose the number of neighbors, 
𝑘
.
Run the search and inspect the results.
Compare search latency and retrieval quality.
Document Ingestion
Enter a document title.
Paste or upload supported text.
Submit the document for processing.
Wait for chunking and embedding generation.
Verify that the chunks have been indexed.
Ask AI
Ingest at least one document.
Enter a question about the document content.
Submit the question.
Let the application retrieve relevant chunks and generate an answer.
Inspect the retrieved context if the interface exposes it.
🌐 REST API

If your application exposes a REST API, document its implemented routes in this section.

The following is an illustrative API reference based on the functionality described for this project. Keep only the routes that exist in your Python implementation.

Method	Endpoint	Description
GET	/search	Search for nearest vectors
POST	/insert	Insert a vector
DELETE	/delete/{id}	Delete a vector
GET	/items	List stored vectors
GET	/benchmark	Compare search algorithms
GET	/hnsw-info	Inspect HNSW index information
GET	/stats	Retrieve database statistics
POST	/doc/insert	Embed and index a document
GET	/doc/list	List indexed documents
DELETE	/doc/delete/{id}	Delete a document or chunk
POST	/doc/ask	Retrieve context and generate an answer
GET	/status	Check application and model status
Example: Vector Search

For an application implementing a compatible endpoint:

bash
curl "http://localhost:8080/search?v=0.1,0.2,0.3&k=5&algo=hnsw"


The vector dimensionality and supported parameters must match your API implementation.

Example: Ask AI

For an application implementing the example RAG endpoint:

bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'


Replace the endpoint and payload with the actual API contract if they differ.

📈 Benchmarking and Performance Evaluation

A vector database should be evaluated on more than raw search speed. VectorForge can be used to explore the trade-offs between latency, recall, memory, and index construction time.

Useful metrics include:

Metric	What It Measures
Query latency	Time needed to retrieve nearest neighbors
Throughput	Queries processed per second
Recall@k	Agreement with exact top-
𝑘
 neighbors
Index build time	Time needed to construct the index
Memory usage	Memory used by vectors and indexes
Dataset size	Number of vectors being indexed
Dimensionality	Number of values in each vector
Recommended Benchmark Procedure
Use the same dataset and query vectors for every algorithm.
Use the same 
𝑘
 and equivalent distance definitions.
Warm up each algorithm before collecting measurements.
Run multiple trials and report median or percentile latency.
Compare HNSW results with Brute Force ground truth.
Record index parameters, dataset size, dimensionality, and hardware.
Separate index construction time from query latency.

No benchmark numbers are included here because results should come from actual experiments on the target implementation and hardware.

🗂️ Project Structure

The structure below is illustrative. Adjust it to match your repository.

text
vector-forge/
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── index.html               # Web UI, if applicable
├── src/
│   ├── vector_db.py         # Unified vector database interface
│   ├── hnsw_index.py        # USearch integration
│   ├── kd_tree.py           # KD-Tree implementation
│   ├── brute_force.py       # Exact nearest-neighbor search
│   ├── embeddings.py        # Embedding model integration
│   ├── document_store.py    # Document ingestion and chunking
│   ├── rag.py               # Retrieval and generation pipeline
│   └── api.py               # REST API routes
├── docs/
│   └── images/              # README screenshots
└── tests/                   # Automated tests, if available

🧰 Technology Stack
Technology	Role
Python	Core application logic
USearch	HNSW-based vector indexing
KD-Tree	Spatial partitioning and nearest-neighbor search
NumPy, if used	Numerical vector operations
Ollama	Local model inference
nomic-embed-text	Text embedding model
llama3.2	Example generation model
PCA	Dimensionality reduction for visualization
Web framework	UI and API serving
🛠️ Troubleshooting
Problem	Suggested Fix
ModuleNotFoundError	Activate the virtual environment and install dependencies.
USearch installation fails	Check Python version, platform support, and the official installation instructions.
Ollama is offline	Start Ollama and verify that its local API is reachable.
Model not found	Run ollama pull for the required model.
Embedding request fails	Verify the model name, request format, and Ollama endpoint.
Search results are unexpected	Check dimensionality, metric configuration, vector normalization, and ranking order.
HNSW misses neighbors	Adjust index/search parameters and measure Recall@k against Brute Force.
KD-Tree is slow	Check dimensionality and pruning effectiveness.
LLM response is slow	Check hardware resources and consider a smaller compatible model.
Port is already in use	Configure an available port or stop the conflicting process.
🔐 Privacy and Limitations
Local model inference can keep embeddings and generation on your machine when all configured services run locally.
Confirm network behavior before using sensitive data; local inference alone does not guarantee that every part of the application is offline.
HNSW is approximate and may miss exact nearest neighbors.
PCA projections can distort high-dimensional relationships.
RAG answers may be inaccurate or unsupported by the retrieved context.
Persistence, concurrent access, authentication, access control, and production reliability depend on the actual implementation.

VectorForge is intended primarily for educational use and experimentation. Production deployment requires additional testing and appropriate operational safeguards.

🎯 Learning Objectives

By exploring VectorForge, you can learn how to:

Represent text and other data as numerical embeddings.
Understand nearest-neighbor search and distance calculations.
Compare exact and approximate retrieval.
Explore graph-based indexing with HNSW.
Integrate USearch into a Python application.
Visualize embeddings using PCA.
Build a semantic document retrieval pipeline.
Combine local embedding models with LLM-based generation.
Evaluate search latency, recall, and memory usage.
Connect a vector engine to a web interface and REST API.
🛣️ Future Improvements

Potential extensions include:

Persistent vector storage and index serialization.
Metadata filtering and hybrid keyword/vector search.
Configurable HNSW parameters and automated tuning.
Batch document ingestion and parallel embedding generation.
Additional embedding models and distance metrics.
Recall-versus-latency benchmark visualizations.
Automated RAG retrieval and answer-quality evaluation.
Tests for insertion, deletion, indexing, and search correctness.
Docker-based setup and deployment.
Improved document provenance and source citations.
🤝 Contributing

Contributions, bug reports, and suggestions are welcome.

Fork the repository.
Create a feature branch.
Implement your changes and add tests where appropriate.
Submit a pull request describing the changes.

For bug reports, include reproducible steps. For performance changes, include benchmark methodology and environment details.

📄 License

This project uses the MIT License if that is the license included in the repository. See the LICENSE file for the applicable terms.

⭐ Support the Project

If you find VectorForge useful for learning vector databases, semantic search, HNSW, or RAG, consider giving the repository a star ⭐.

Built to explore vector search under the hood — one embedding at a time.
