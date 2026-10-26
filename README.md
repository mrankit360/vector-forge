## VectorDB — Build a Vector Database from Scratch in Python

A lightweight, educational Vector Database built with Python, featuring multiple nearest-neighbor search algorithms, a web-based UI, semantic document search, and a Retrieval-Augmented Generation (RAG) pipeline powered by local LLMs through Ollama.

Explore how vector databases work under the hood by comparing HNSW, KD-Tree, and Brute Force search side by side, visualizing vector embeddings, and building a complete local RAG application.

Learn the core concepts behind vector databases such as Pinecone, Weaviate, Chroma, and Milvus — from vector indexing and similarity search to document retrieval and LLM-powered question answering.

✨ Features
- Multiple Search Algorithms — Compare HNSW, KD-Tree, and Brute Force nearest-neighbor search.
- USearch-Powered HNSW — Use USearch for efficient approximate nearest-neighbor indexing.
- Multiple Distance Metrics — Cosine similarity, Euclidean distance, and Manhattan distance, where supported by the implementation.
- Interactive Web UI — Search vectors, inspect results, visualize embeddings, and explore algorithm performance.
- Vector Visualization — Project high-dimensional vectors into 2D using PCA to explore semantic clusters.
- Real Document Embeddings — Convert text into dense vector representations using Ollama and nomic-embed-text.
- Document Chunking — Split long documents into smaller, overlapping chunks for more relevant retrieval.
- Local RAG Pipeline — Retrieve relevant document chunks and use a local language model to generate grounded answers.
- REST API — Integrate vector search, document ingestion, and question answering into other applications.
- Local-First Architecture — Run embeddings, retrieval, and generation locally without relying on a hosted AI API.
  
🧠 What Is a Vector Database?

Traditional databases primarily search structured data using exact matches, indexes, and filters. A vector database stores numerical representations of data called embeddings and retrieves items based on their similarity in vector space.

For example, a traditional keyword search may not recognize that these phrases are related:

"How does a binary tree work?"
"Understanding tree data structures"
"Binary tree traversal algorithms"

A semantic search engine compares their embeddings to find conceptually related content, even when the wording differs.

This project demonstrates the fundamentals behind that process, from storing vectors and calculating distances to approximate nearest-neighbor search and retrieval-augmented generation.

🏗️ Architecture

The project has two main workflows: vector search and document question answering.

1. Vector Search
text
             Search Query
                  |
                  v
          Query Vector
                  |
                  v
       +--------------------+
       |  VectorDB Engine   |
       +--------------------+
          /       |       \
         v        v        v
     Brute     KD-Tree   HNSW
     Force                |
          \       |       /
           \      |      /
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
          Ollama Embedding Model
             nomic-embed-text
                    |
                    v
             Vector Index
               (USearch)
                    |
                    |
              Your Question
                    |
                    v
          Embed the Question
                    |
                    v
          Semantic Retrieval
                    |
                    v
           Relevant Text Chunks
                    |
                    v
             Local LLM
               (Ollama)
                    |
                    v
              Final Answer


The embedding model converts text into numerical vectors. The vector index retrieves semantically relevant chunks, which are passed to the language model as context for generating an answer.

🔍 Search Algorithms

One of the main goals of this project is to demonstrate how different nearest-neighbor algorithms behave on the same dataset.

Algorithm	Search Type	Main Advantage	Main Limitation
Brute Force	Exact	Simple and reliable baseline	Scales linearly with dataset size
KD-Tree	Exact, with pruning	Effective for suitable low-dimensional data	Performance degrades in high dimensions
HNSW (USearch)	Approximate	Fast retrieval on many high-dimensional workloads	Uses additional index memory and may miss true nearest neighbors
Brute Force Search

Brute Force calculates the distance between the query and every stored vector, then returns the nearest results.

Provides an exact baseline for supported distance metrics.
Easy to understand and implement.
Useful for validating approximate search results.
Search cost grows with the number of vectors and their dimensionality.


KD-Tree

A KD-Tree recursively partitions vector space along selected dimensions. Search explores promising branches and prunes branches that cannot improve the current nearest-neighbor results.

Can perform well on low-dimensional datasets.
Provides exact nearest-neighbor results when implemented with correct pruning.
May lose its advantage as dimensionality increases.

In high-dimensional spaces, pruning often becomes less effective, and search can approach a linear scan.

HNSW with USearch

HNSW stands for Hierarchical Navigable Small World. It builds a multilayer proximity graph in which upper layers provide long-range navigation and lower layers enable more detailed local search.

USearch provides an optimized implementation of this indexing approach.

Efficient approximate nearest-neighbor retrieval.
Well suited to many embedding-search workloads.
Supports configurable index behavior and distance metrics, subject to the selected USearch API and configuration.
Trades some retrieval accuracy and additional memory usage for speed.

Actual performance depends on dataset size, dimensionality, index parameters, hardware, and query distribution.

Important: HNSW is an approximate search algorithm. Use Brute Force as a reference when evaluating recall and retrieval quality.

📐 Distance Metrics

Distance metrics determine how the similarity between two vectors is measured.

Cosine Similarity

Measures the angle between two vectors and is commonly used for semantic embeddings.


Higher cosine similarity indicates greater directional similarity. If the application converts cosine similarity into cosine distance, the ranking convention is reversed: lower distance is better.

Euclidean Distance

Measures straight-line distance between vectors.

Lower values indicate closer vectors.

Manhattan Distance

Measures the sum of the absolute differences between corresponding dimensions.

Lower values indicate closer vectors.

Implementation note: Confirm that all three metrics are supported consistently across your algorithms. USearch index metric configuration and the way results are ranked should match the metric selected in the UI.
📊 Vector Visualization

The web interface includes a 2D visualization of vector space using Principal Component Analysis (PCA), if enabled in your implementation.

PCA reduces high-dimensional vectors to two principal components while retaining as much variance as possible in the projected data.

The visualization can help you explore:

How related concepts form clusters.
How different categories are distributed in vector space.
Where a query sits relative to stored vectors.
Which vectors are returned by a nearest-neighbor search.

Note: A 2D PCA projection is only a visual approximation of the original space. Points that look close on the chart are not necessarily the nearest neighbors in the full-dimensional space.

## 🤖 Local AI with Ollama

This project use Ollama to run embedding and language models locally.

Embedding Model

nomic-embed-text converts text into dense vector embeddings for semantic search.

Generation Model

llama3.2 generates responses using the document chunks retrieved by the RAG pipeline.

The embedding model and generation model serve different purposes:

Component	Responsibility
nomic-embed-text	Converts documents and questions into vectors
USearch / HNSW	Finds the nearest stored vectors
Retrieval pipeline	Selects relevant document chunks
llama3.2	Generates a response using retrieved context

The embedding dimensionality depends on the selected model and configuration. For the standard nomic-embed-text setup, embeddings are typically 768-dimensional.

## 📚 RAG Pipeline

Retrieval-Augmented Generation combines semantic search with text generation. Instead of asking an LLM to answer using only its pretrained knowledge, the application retrieves relevant information from your own documents and supplies it as context.

How It Works
Ingest: Provide a document or paste text into the application.
Chunk: Split long text into smaller, overlapping chunks.
Embed: Generate an embedding for each chunk.
Index: Store each embedding alongside its text and metadata.
Retrieve: Embed the user's question and find the nearest chunks.
Generate: Send the question and retrieved context to the local language model.
Respond: Display the generated answer and, where supported, the retrieved context.
Example

Suppose you insert notes about dynamic programming and ask:

What is dynamic programming, and when should it be used?

The system embeds the question, searches the document index, retrieves relevant passages, and provides those passages to the LLM as context.

The quality of the answer depends on document quality, chunking, embedding quality, retrieval settings, and the language model.

Grounding note: Providing retrieved context does not guarantee that an LLM will only use that context. For more reliable answers, instruct the model to acknowledge missing information and inspect the retrieved passages or citations.

## ⚙️ Prerequisites

Install the following software before running the project.

Python 3.12+ — Adjust this version to match your project's actual requirements.
Git — To clone the repository.
Ollama — Required for local embeddings and RAG generation.
Required Python packages — Installed from the project's requirements.txt.

USearch supports multiple operating systems and installation methods. Follow its official installation instructions if a platform-specific dependency issue occurs.

🚀 Getting Started
Step 1 — Clone the Repository
bash
git clone https://github.com/mrankit360/vector-forge.git
cd vector-forge

VectorDB with the actual repository name if different.

Step 2 — Create a Virtual Environment

Windows PowerShell

powershell
python -m venv myenv
myenv\Scripts\Activate.ps1


Linux / macOS

bash
python3 -m venv myenv
source myenv/bin/activate


If PowerShell prevents virtual-environment activation, use the appropriate execution-policy guidance for your environment or run the activation script from Command Prompt instead.

Step 3 — Install Dependencies

If the repository contains a requirements.txt file:

bash
python -m pip install --upgrade pip
pip install -r requirements.txt


Step 4 — Install Ollama and Download Models

Download Ollama from ollama.com.

Pull the embedding model:

bash
ollama pull nomic-embed-text


Pull the generation model:

bash
ollama pull llama3.2


Check the installed models:

bash
ollama list


Ollama generally runs as a background service after installation. If it is not running, start it with:

bash
ollama serve


Avoid starting a second server if Ollama is already running.

Step 5 — Start the Application

Run the entry point used by your project. For example, if the application entry point is main.py:

bash
uvicorn app.main:app --reload

Open the local URL printed by the application in your browser.

The commands above are setup templates. The exact entry-point filename, web server command, and URL should match the actual Python repository.
🖥️ Using the Application

The following sections describe the intended workflow. Rename the tabs or controls to match the actual UI.

Vector Search
Enter a query or select a sample vector.
Choose the search algorithm: HNSW, KD-Tree, or Brute Force.
Select the distance metric supported by the application.
Choose the number of nearest neighbors, 
𝑘
.
Run the search and inspect the results.
Compare latency and result quality across algorithms.
Document Ingestion
Enter a document title.
Paste or upload supported text.
Submit the document for processing.
Allow the application to split the document into chunks and generate embeddings.
Confirm that the document has been indexed.
Ask AI
Ingest at least one document.
Enter a question about the stored content.
Submit the question to the RAG pipeline.
Inspect the generated response.
Review the retrieved chunks, if the UI exposes them.
🌐 REST API

If your Python application exposes a REST API, document its actual endpoints here.

An example API reference for a project with equivalent functionality might look like this:

Method	Endpoint	Purpose
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
POST	/doc/ask	Run RAG question answering
GET	/status	Check model and application status

These are example routes, not verified routes from your Python implementation. Remove any endpoint that is not implemented and document the exact request parameters, JSON schemas, response structures, and error codes.

Example Search Request

For an application implementing a compatible search endpoint:

bash
curl "http://localhost:8080/search?v=0.1,0.2,0.3&k=5&algo=hnsw"


The vector dimensionality, metric parameters, and accepted algorithm names must match your API implementation.

Example RAG Request

For an application implementing the example document-question endpoint:

bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'


On Windows PowerShell, curl.exe can be used explicitly if curl resolves to a PowerShell alias.

📈 Benchmarking and Evaluation

Comparing algorithms is more meaningful when performance and accuracy are measured separately.

Useful measurements include:

Query latency: Time taken to retrieve the nearest neighbors.
Throughput: Number of queries processed per second.
Recall@k: Fraction of the exact top-
𝑘
 neighbors also found by approximate search.
Index build time: Time required to construct the index.
Memory usage: Memory consumed by vectors and index structures.
Dataset size: Number of vectors and their dimensionality.
Suggested Benchmark Methodology
Generate or load the same dataset for each algorithm.
Use the same query vectors, 
𝑘
, and compatible distance metric.
Run warm-up queries before recording measurements.
Execute multiple repetitions and report median or percentile latency.
Compare approximate HNSW results with Brute Force ground truth.
Record the dataset size, dimensions, hardware, and index configuration.
Separate index construction time from query latency.

Avoid publishing benchmark numbers without running the experiments. Performance depends on the workload, hardware, vector distribution, and index parameters.

🗂️ Project Structure

Update this example to reflect your actual repository layout.

text
VectorDB/
├── main.py                 # Application entry point (example)
├── index.html              # Web interface (if applicable)
├── requirements.txt        # Python dependencies (if applicable)
├── README.md               # Project documentation
├── src/
│   ├── vector_db.py        # Vector storage and search interface
│   ├── hnsw_index.py       # USearch integration
│   ├── kd_tree.py          # KD-Tree implementation
│   ├── brute_force.py      # Exact nearest-neighbor search
│   ├── embeddings.py       # Embedding model integration
│   ├── document_store.py   # Document ingestion and chunking
│   ├── rag.py              # Retrieval and generation pipeline
│   └── api.py              # API routes (if applicable)
└── tests/                  # Automated tests (if applicable)


The directory and file names above are illustrative. Keep only the files and modules that exist in your repository.

🧩 Technology Stack
Technology	Role
Python	Core application and orchestration
USearch	HNSW-based vector indexing
KD-Tree	Spatial partitioning for nearest-neighbor search
NumPy or equivalent	Numerical vector operations, if used
Ollama	Local embedding and text generation
nomic-embed-text	Text embedding model
llama3.2	Example local generation model
PCA	Dimensionality reduction for visualization, if implemented
Web framework	UI/API serving; specify the framework used in this project
🛠️ Troubleshooting
Problem	What to Check
ModuleNotFoundError	Activate the virtual environment and install the project dependencies.
USearch installation fails	Check Python version, platform compatibility, and the official installation documentation.
Ollama is offline	Start Ollama and verify that its local API is reachable.
Model not found	Pull the required model using ollama pull.
Embedding requests fail	Verify the embedding model name and the Ollama endpoint configuration.
Search returns unexpected results	Check vector dimensionality, metric configuration, normalization, and result ordering.
HNSW misses expected neighbors	Tune the index/search parameters and evaluate Recall@k against Brute Force.
KD-Tree search is slow	Check dimensionality and pruning behavior; KD-Trees often struggle with high-dimensional data.
LLM responses are slow	Check hardware resources and consider a smaller compatible generation model.
Port is already in use	Configure a different available port or stop the conflicting process.
🔐 Privacy and Limitations
Local inference can keep document processing on your machine when all configured services run locally.
This does not automatically guarantee that the application never sends data elsewhere; verify network calls and integrations.
HNSW is approximate, so it may not return the exact nearest neighbors.
PCA visualizations can distort high-dimensional relationships.
RAG responses can contain incorrect or unsupported claims.
Persistence, concurrency, authentication, access control, and production deployment depend on the actual implementation.

This project is intended primarily for learning and experimentation. Do not assume production-grade durability, security, or scalability without additional testing.

🎯 Learning Objectives

By exploring this project, you can learn how to:

Represent text and other data as numerical embeddings.
Implement nearest-neighbor search and compare algorithmic trade-offs.
Understand graph-based approximate search through HNSW.
Integrate USearch into a Python application.
Build a semantic document retrieval pipeline.
Combine local embedding models with LLM-based generation.
Evaluate search speed, recall, and memory consumption.
Connect a vector search engine to a web UI and REST API.
🚀 Future Improvements

Potential directions for extending the project include:

Persistent storage and index serialization.
Metadata filtering and hybrid keyword/vector search.
Configurable HNSW parameters and automated tuning.
More embedding models and distance metrics.
Batch ingestion and parallel embedding generation.
Recall-versus-latency benchmark visualizations.
Automated evaluation of RAG retrieval and answer quality.
Automated tests for indexing, deletion, persistence, and search correctness.
Docker-based setup and deployment.
🤝 Contributing

Contributions, bug reports, and suggestions are welcome.

Fork the repository.
Create a feature branch.
Make your changes and add tests where appropriate.
Submit a pull request describing the changes.

Please include reproducible steps for bug reports and benchmark details for performance-related changes.

📄 License

This project is distributed under the MIT License if the repository includes that license. See the LICENSE file for the applicable terms.

⭐ Support the Project

If you find this project useful for learning vector databases, semantic search, or RAG systems, consider giving the repository a star.

Built to explore the internals of vector search — one embedding at a time.
