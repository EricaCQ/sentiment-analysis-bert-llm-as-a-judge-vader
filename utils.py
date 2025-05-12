import os
from openai import OpenAI
from dotenv import load_dotenv

# # Load environment variables (API KEY) from a .env file
load_dotenv()

# Creates an OpenAI client using the API key stored in the environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

 # Sends a prompt to the OpenAI Chat API and returns the response.

  #Returns:
    #- str: The content of the response from the model, or a default message on error.
    
def call_openai(prompt, model="gpt-3.5-turbo", max_tokens=50):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in OpenAI call {e}")
        return "I am not sure how to classify, need human annotator."
