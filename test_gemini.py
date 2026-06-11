import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("GEMINI_API_KEY is missing")
else:
    print("Gemini API key loaded successfully")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Return only this text: Gemini API is working"
)

print(response.text)