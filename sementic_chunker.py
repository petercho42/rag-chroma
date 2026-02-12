import re


def semantic_chunker(text, max_sentences=3, overlap_sentences=1):
    # Split text into sentences using regex
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks = []

    i = 0
    while i < len(sentences):
        # Take a slice of sentences
        chunk = " ".join(sentences[i : i + max_sentences])
        chunks.append(chunk)
        # Slide the window forward, but subtract the overlap
        i += max_sentences - overlap_sentences

    return chunks


# Test it with a paragraph about RAG or your favorite tech topic
sample_text = "RAG is great. It uses vectors. Vectors are math. Math is hard. But AI helps. AI is cool."
print(semantic_chunker(sample_text, 3, 1))
