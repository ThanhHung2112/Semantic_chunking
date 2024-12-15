from semantic_chunking import SemanticChunker

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
test_semantic_chunker()