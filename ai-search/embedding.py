import os
from dotenv import load_dotenv

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

aoai_endpoint = os.getenv('AOAI_ENDPOINT')
aoai_api_key = os.getenv('AOAI_API_KEY')
aoai_api_version = os.getenv('API_VERSION')
aoai_deployment = os.getenv('DEPLOYMENT_NAME')

ai_search_endpoint = os.getenv('AI_SEARCH_ENDPOINT')
ai_search_key = os.getenv('AI_SEARCH_KEY')
ai_search_index_name = os.getenv('AI_SEARCH_INDEX')


# Create Embeddiong and Vector store
embeddings = AzureOpenAIEmbeddings(
  azure_deployment=aoai_deployment,
  azure_endpoint=aoai_endpoint,
  api_key=aoai_api_key,
  api_version=aoai_api_version
)

vector_store = AzureSearch(
  azure_search_endpoint=ai_search_endpoint,
  azure_search_key=ai_search_key,
  index_name=ai_search_index_name,
  embedding_function=embeddings.embed_query
)


