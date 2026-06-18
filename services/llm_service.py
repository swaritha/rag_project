from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def generate_answer(question, contexts):

    context_text = "\n".join(contexts)

    prompt = f"""
    Context:
    {context_text}

    Question:
    {question}

    Answer:
    """

    result = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False
    )

    return result[0]["generated_text"]