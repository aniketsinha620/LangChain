from dotenv import load_dotenv
import os

def main():
    print("Hello from langchain-course!")
    print(os.environ.get("GROQ_API_KEY"))


load_dotenv()
if __name__ == "__main__":
    main()
