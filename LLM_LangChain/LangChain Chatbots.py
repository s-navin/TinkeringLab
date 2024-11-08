
from langchain_huggingface import HuggingFaceEndpoint

# Set your Hugging Face API token 
huggingfacehub_api_token = your_huggingfacehub_access_token

# Define an LLM using the Falcon-7B instruct model from Hugging Face, which has the ID: 'tiiuae/falcon-7b-instruct'
llm = HuggingFaceEndpoint(repo_id='tiiuae/falcon-7b-instruct', huggingfacehub_api_token=huggingfacehub_api_token)

# Predict the words following the text in question
question = 'Whatever you do, take care of your shoes'
output = llm.invoke(question)

print(output)
