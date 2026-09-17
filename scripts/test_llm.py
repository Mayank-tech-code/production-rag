from app.generation.llm import get_llm_client


client = get_llm_client()


response = client.chat_completion(
    messages=[
        {
            "role": "user",
            "content": "Explain what React is in two sentences."
        }
    ],
    max_tokens=100,
    temperature=0.1,
)


print(response.choices[0].message.content)