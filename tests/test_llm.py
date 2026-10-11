from src.llm import generate_answer


prompt = """
Explain what magnesium alloys are in simple terms.
Give the answer in 3-4 sentences.
"""


answer = generate_answer(prompt)


print("\n" + "=" * 70)
print("GEMINI TEST")
print("=" * 70)

print("\nPrompt:")
print(prompt)

print("\nGemini response:")
print(answer)