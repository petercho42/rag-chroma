def augment_prompt(query, context_chunks):
    # Join all retrieved chunks into one big block of text
    context_text = "\n\n".join(context_chunks)

    prompt = f"""
    You are a professional assistant. Use the provided Context to answer the User Question.
    If the answer is not contained within the Context, honestly state that you do not know.
    Do not use outside knowledge.

    ### Context:
    {context_text}

    ### User Question:
    {query}

    ### Answer:
    """
    return prompt
