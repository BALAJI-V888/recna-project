import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

client = Groq(api_key=GROQ_API_KEY)


weights = {
    "introduction": 0.10,
    "methodology": 0.35,
    "discussion": 0.35,
    "conclusion": 0.20
}