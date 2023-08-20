# pip install -q langchain openai chromadb
# pip install tiktoken

from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

os.environ["OPENAI_API_KEY"] = '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'
# Load the documents
loader = CSVLoader(file_path='builderdatacsv.csv')

# Create an index using the loaded documents
index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])

chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

query = "Tell me about a property in Sushant Lok 1."
response = chain({"question": query})
print(response['result'])
