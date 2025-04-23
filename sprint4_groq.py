from groq import Groq

# Initialize Groq client with your API key
client = Groq(api_key="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")

def call_groq_llm(context, question):
    """Call Groq's LLaMA3-8B model with context-based prompt."""
    prompt = f"You are a helpful assistant. Answer the question based only on the context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    max_tokens = 4096
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a PDF chatbot assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content
