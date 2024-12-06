# CS646 FinSMART

CS646 FinSMART is a structured project for financial document retrieval, reranking, and response generation. It provides tools and workflows to process financial datasets, retrieve relevant documents, rerank them effectively, and generate meaningful responses using hybrid and generation-based approaches.

---

## **Project Structure**

The repository is organized into the following directories and files:
```
| financerag/                     # Main module directory
│   ├── common/                   # Common utility functions
│   ├── generate/                 # Code for response generation
│   ├── rerank/                   # Code for reranking retrieved documents
│   └── retrieval/                # Code for document retrieval
├── data/                         # Dataset storage folder
├── results/                      # Results folder
├── pre_retrieval.ipynb           # Script for pre-retrieval
├── retreival.ipynb                  # Code for first-stage retrieval and reranking
├── prompt.json                   # Configuration for pre-retrieval prompts
├── requirements.txt              # Python dependencies
├── hybrid.ipynb                  # Code for hybrid score computation
└── generation.ipynb              # Code for response generation
```
### **Key Components**

1. **`financerag/`**
   - **`common/`**: Contains shared utility functions (e.g., logging, preprocessing).
   - **`generate/`**: Implements methods for response generation using retrieved documents.
   - **`rerank/`**: Handles reranking of the retrieved documents to improve result quality.
   - **`retrieval/`**: Responsible for document retrieval from the corpus using predefined models.

2. **`data/`**
   - Stores the datasets required for document retrieval and processing.

3. **`results/`**
   - Directory for saving results from retrieval, reranking, and response generation steps.

4. **`pre_retrieval.ipynb`**
   - Jupyter Notebook for performing preprocessing steps and initializing retrieval tasks.

5. **`retreival.ipynb`**
   - Script for executing the first stage of retrieval and reranking the documents.

6. **`prompt.json`**
   - JSON configuration file containing prompts for pre-retrieval tasks.

7. **`hybrid.ipynb`**
   - Notebook for calculating hybrid scores that combine different retrieval strategies.

8. **`generation.ipynb`**
   - Jupyter Notebook for performing response generation.

9. **`requirements.txt`**
   - Lists all Python dependencies needed for the project.

---

## **Setup and Installation**

### **Prerequisites**
- Python 3.11 or later
- pip (Python package manager)

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/CS646_FinSmart.git
   cd CS646_FinSmart
