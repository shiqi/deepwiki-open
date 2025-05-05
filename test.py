from google import genai
from google.genai import types
from google.generativeai import GenerativeModel
import os
from openai import OpenAI




# Configure the API key.  It's best to set this as an environment variable.
# If you don't have it set, you can set it directly, but that's less secure.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # WARNING: Setting the API key directly in the code is NOT recommended for production use.
    # It's better to set it as an environment variable.
    # Example (in your terminal): export GOOGLE_API_KEY="YOUR_API_KEY"
    # Then, the os.environ.get("GOOGLE_API_KEY") line above will work.
    GOOGLE_API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
    print(
        "WARNING: Setting the API key directly in the code.  Use an environment variable for security."
    )



def generate_embeddings(texts, model_name="gemini-embedding-003-07"):
    """
    Generates embeddings for a list of texts using a specified Gemini embedding model.

    Args:
        texts: A list of strings to generate embeddings for.
        model_name: The name of the Gemini embedding model to use.
            Defaults to "gemini-embedding-003-07".

    Returns:
        A list of embeddings.  Each embedding is a list of floats.
        Returns None and prints an error if there's a problem.
    """
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        for text in texts:
            result = client.models.embed_content(
                model="text-embedding-004",
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=256),
            )
            # The pipeline expects a list of embeddings, each with a 'data' attribute
            embedding = {
                'data': result.embedding
            }
            embeddings.append(embedding)
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None



def main():
    """
    Main function to demonstrate generating embeddings with Gemini.
    """
    texts = [
        "This is the first text to embed.",
        "Here is a second sentence for embedding.",
        "A third example, slightly longer, to show how it handles different lengths.",
    ]

    print(f"Generating embeddings for the following texts using model 'gemini-embedding-003-07':")
    for text in texts:
        print(f"- \"{text}\"")

    embeddings = generate_embeddings(texts)

    if embeddings:
        print("\nEmbeddings:")
        for i, embedding in enumerate(embeddings):
            print(f"Text {i+1}:")
            print(f"  Length: {len(embedding['data'])}")  # Print the dimensionality
            print(f"  First 5 values: {embedding['data'][:5]}...")  # Print a snippet for brevity

        # Example of using the embeddings (cosine similarity):
        from numpy import dot
        from numpy.linalg import norm

        def cosine_similarity(a, b):
            """Calculates the cosine similarity between two vectors."""
            return dot(a, b) / (norm(a) * norm(b))

        if len(embeddings) > 1:
            similarity_1_2 = cosine_similarity(embeddings[0]['data'], embeddings[1]['data'])
            print(f"\nCosine similarity between text 1 and 2: {similarity_1_2:.4f}")
    else:
        print("\nFailed to generate embeddings.")



if __name__ == "__main__":
    client = OpenAI(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = client.embeddings.create(
        input="Your text string goes here",
        model="text-embedding-004"
    )

    print(response)
    #main()
