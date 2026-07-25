import os
from api import main_api
from dotenv import load_dotenv

def main():
    print("Hello from langchain-course!")

    groq_response=main_api("groq")
    # groq_response=main_api("ollama")
    print("Response from groq chain:\n\n", groq_response)




load_dotenv()
if __name__ == "__main__":
    main()
