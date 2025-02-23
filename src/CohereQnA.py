import os
from urllib import response
import cohere
from dotenv import load_dotenv
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.llms import Cohere 

load_dotenv()

# Initialize the Cohere client
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY environment variable not set")

client = cohere.Client(api_key)

from pydantic import SecretStr

llm = Cohere(
    cohere_api_key=SecretStr(api_key),
    model="command-light",
    temperature=0
)

embeddings = CohereEmbeddings( model="embed-english-v3.0",user_agent="langchain")

#RAG-based QnA
from langchain_community.document_loaders import TextLoader


PROJECT_ROOT = "<Path to the project root directory>"
file_path = os.path.join(PROJECT_ROOT, "data/globalcorp_hr_policy.txt")
loader = TextLoader(file_path)
documents = loader.load()

from langchain_chroma import Chroma

persist_directory = "local_vectorstore"
collection_name = "hrpolicy"

vectorstore = Chroma(
    persist_directory=os.path.join(PROJECT_ROOT, "data", persist_directory),
    collection_name=collection_name,
    embedding_function=embeddings,
)

from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore


local_store = "local_docstore"
local_store = LocalFileStore(os.path.join(PROJECT_ROOT, "data", local_store))
docstore = create_kv_docstore(local_store)


from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Creates small, granular chunks that are good for precise matching. These are what get embedded and used for the initial similarity search.
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

#Creates larger chunks that preserve more context. When a child chunk is retrieved, the system can fetch its corresponding parent chunk to provide more context.
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# run only once
#vectorstore.persist()
retriever.add_documents(documents, ids=None)


from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

response = qa({"query": "What is the allocated budget for communication initiatives?"})
print(response)