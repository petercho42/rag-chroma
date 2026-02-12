import re


def semantic_chunker_pro(text, source_name, max_sentences=3, overlap_sentences=1):
    sentences = re.split(r"(?<=[.!?]) +", text)
    processed_chunks = []

    i = 0
    chunk_id = 0
    while i < len(sentences):
        content = " ".join(sentences[i : i + max_sentences])

        # This is the format Vector DBs expect
        processed_chunks.append(
            {
                "id": f"{source_name}_{chunk_id}",
                "text": content,
                "metadata": {
                    "source": source_name,
                    "index": chunk_id,
                    "sentence_count": len(sentences[i : i + max_sentences]),
                },
            }
        )

        i += max_sentences - overlap_sentences
        chunk_id += 1

    return processed_chunks


# Try it out
data = semantic_chunker_pro(
    "RAG is the future. It solves hallucinations. Vectors represent meaning.",
    "AI_Blog_01",
)
print(data[0])
