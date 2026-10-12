# Materials Research Assistant

A Streamlit-based research assistant that uses Retrieval-Augmented Generation (RAG) to help users explore and analyze materials science research papers.

## Features

- Ask questions about uploaded research papers.
- Generate research paper summaries.
- Compare information across papers.
- Retrieve relevant passages using semantic and keyword-based search.
- Rerank retrieved passages before generating answers.
- Display source papers and relevant page numbers.
- Explore research through the application's analysis tools.

## Technology Stack

- Python
- Streamlit
- Google Gemini API
- ChromaDB
- Sentence Transformers
- BM25
- Pytest

## Project Structure

```text
Materials-Research-Assistant/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
├── tests/
└── data/  # Local data; excluded from Git
```

## Requirements

- Python 3.11
- A Google Gemini API key
- Packages listed in `requirements.txt`

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Materials-Research-Assistant
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

In PowerShell, set the key for the current terminal session:

```powershell
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Replace the placeholder with your own API key. Never commit the key to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## How to Use

1. Launch the application.
2. Upload your research papers through the application's upload interface.
3. Wait for the papers to be processed and indexed.
4. Choose a question-answering, summarization, comparison, or other available analysis tool.
5. Review the generated response and its associated sources.
6. Ask follow-up questions to explore the research further.

## Data and Storage

- The Gemini API key is read from the `GEMINI_API_KEY` environment variable.
- ChromaDB stores the indexed research passages.
- Uploaded documents and the vector database are stored locally according to the application's configuration.
- Local documents, databases, virtual environments, and secrets should not be committed to the repository.

Cloud deployment requires separate configuration for secrets and consideration of persistent document and vector storage.

## Testing

Run the test suite with:

```bash
python -m pytest tests/ -v
```

Review individual test scripts before running them, especially any script that deletes or rebuilds the local vector database.

## Limitations

- Answer quality depends on PDF text extraction and retrieval quality.
- Generated responses should be verified against the original research papers.
- Gemini API usage may be subject to quotas and rate limits.
- Local research data does not automatically transfer to a cloud deployment.

## Future Improvements

- Deploy the application publicly.
- Improve PDF extraction and source presentation.
- Add persistent cloud storage.
- Expand end-to-end testing.


