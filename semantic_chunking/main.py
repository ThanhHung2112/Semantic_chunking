import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class SemanticChunker:
    def __init__(self, model_name='all-MiniLM-L6-v2', max_chunk_size=265, similarity_threshold=0.3):
        """
        Initialize semantic chunker with a sentence transformer model

        Args:
            model_name: Sentence Transformer model to use
            max_chunk_size: Maximum number of words per chunk
            similarity_threshold: Cosine similarity threshold for chunk splitting
        """
        self.model = SentenceTransformer(model_name)
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold

    def semantic_chunk(self, text: str) -> List[str]:
        """
        Chunk text semantically using embedding-based similarity

        Args:
            text: Input text to chunk

        Returns:
            List of semantic chunks
        """
        # Split text into sentences
        sentences = self._split_into_sentences(text)

        # If text is too short, return as is
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # Embed sentences
        sentence_embeddings = self.model.encode(sentences)

        chunks = []
        current_chunk = []
        current_chunk_words = 0

        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            sentence_words = sentence.split()

            # Check if adding this sentence would exceed max chunk size
            if current_chunk_words + len(sentence_words) > self.max_chunk_size:
                # If chunk is not empty, add it and reset
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_chunk_words = 0

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_chunk_words += len(sentence_words)

            # Check semantic similarity with next sentence if exists
            if i < len(sentences) - 1:
                next_embedding = sentence_embeddings[i + 1]
                similarity = self._cosine_similarity(embedding, next_embedding)

                # If similarity is low, it might indicate a semantic break
                if similarity < self.similarity_threshold:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_chunk_words = 0

        # Add remaining chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, handling various punctuation
        """
        import re
        # Use regex to split into sentences, handling abbreviations and titles
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Example usage
def test_semantic_chunker():
    text = """Machine learning is a subset of artificial intelligence that focuses on the use of data and algorithms to imitate the way that humans learn.
    Deep learning is a more specialized version of machine learning that uses neural networks with multiple layers.
    These advanced algorithms can process complex patterns and make intelligent decisions with minimal human intervention.
    The applications of machine learning are vast, ranging from image recognition to natural language processing."""

    chunker = SemanticChunker(model_name='all-MiniLM-L6-v2', max_chunk_size=128, similarity_threshold=0.3)
    chunks = chunker.semantic_chunk(text)

    print("Semantic Chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")
        print(f"Words: {len(chunk.split())}\n")

# Uncomment to test
# test_semantic_chunker()