from typing import Dict, List, Literal, Optional, Union
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

class BM25Processor:
    """
    A processor for BM25-based lexical retrieval that handles query and corpus tokenization,
    initialization of the BM25 model, and computation of relevance scores.
    """

    def __init__(
            self,
            tokenizer: Optional[callable] = None,
            query_prompt: Optional[str] = None,
            doc_prompt: Optional[str] = None,
    ):
        """
        Initializes the BM25Processor.

        Args:
            tokenizer (`Optional[callable]`, optional):
                A function to tokenize input strings. Defaults to `nltk.word_tokenize`.
            query_prompt (`Optional[str]`, optional):
                A prompt to prepend to each query before tokenization. Defaults to `None`.
            doc_prompt (`Optional[str]`, optional):
                A prompt to prepend to each document before tokenization. Defaults to `None`.
        """
        self.tokenizer = tokenizer or word_tokenize  # Default to nltk word tokenizer
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt
        self.bm25: Optional[BM25Okapi] = None  # BM25 model, to be initialized with corpus

    def build_corpus(
            self,
            corpus: Union[
                List[Dict[Literal["title", "text"], str]],
                Dict[Literal["title", "text"], List],
            ],
    ) -> List[List[str]]:
        """
        Tokenizes and processes the corpus for BM25 initialization.

        Args:
            corpus (Union[List[Dict[Literal["title", "text"], str]], Dict[Literal["title", "text"], List]]):
                The corpus to process, either as a list of documents or a dictionary of lists.

        Returns:
            `List[List[str]]`: A list of tokenized documents.
        """
        if isinstance(corpus, dict):
            sentences = [
                (
                    (corpus["title"][i] + " " + corpus["text"][i]).strip()
                    if "title" in corpus
                    else corpus["text"][i].strip()
                )
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (
                    (doc["title"] + " " + doc["text"]).strip()
                    if "title" in doc
                    else doc["text"].strip()
                )
                for doc in corpus
            ]

        if self.doc_prompt is not None:
            sentences = [self.doc_prompt + s for s in sentences]

        return [self.tokenizer(sentence) for sentence in sentences]

    def initialize(self, corpus: List[List[str]]):
        """
        Initializes the BM25 model with the tokenized corpus.

        Args:
            corpus (`List[List[str]]`): A tokenized corpus for BM25 indexing.
        """
        self.bm25 = BM25Okapi(corpus)

    def compute_scores(self, query: str, top_k: Optional[int] = None) -> Dict[int, float]:
        """
        Computes relevance scores for a given query against the corpus.

        Args:
            query (`str`): The input query string.
            top_k (`Optional[int]`, optional): The number of top documents to return. Defaults to all.

        Returns:
            `Dict[int, float]`: A dictionary where keys are document indices and values are relevance scores.
        """
        if not self.bm25:
            raise ValueError("BM25 model has not been initialized. Call `initialize` first.")

        if self.query_prompt is not None:
            query = self.query_prompt + query

        query_tokens = self.tokenizer(query)
        scores = self.bm25.get_scores(query_tokens)

        # Sort scores and return top_k if specified
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]

        return {idx: scores[idx] for idx in sorted_indices}

    def retrieve(
            self,
            queries: Dict[str, str],
            top_k: Optional[int] = None
    ) -> Dict[str, Dict[int, float]]:
        """
        Retrieves the most relevant documents for each query.

        Args:
            queries (`Dict[str, str]`): A dictionary of query IDs and query strings.
            top_k (`Optional[int]`, optional): The number of top documents to return per query. Defaults to all.

        Returns:
            `Dict[str, Dict[int, float]]`: A dictionary where each key is a query ID and the value
            is a dictionary mapping document indices to relevance scores.
        """
        results = {}
        for query_id, query in queries.items():
            results[query_id] = self.compute_scores(query, top_k=top_k)
        return results
