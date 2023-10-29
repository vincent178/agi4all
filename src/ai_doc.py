from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain import hub
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

MODEL = "mistral:latest"

# Load
loader = DirectoryLoader('../../GOAT/engineering-docs/website/docs', glob="**/*.md", use_multithreading=True)
docs = loader.load()

doc_with_error = None
for doc in docs:
    if doc.metadata.get("source") == '../../GOAT/engineering-docs/website/docs/frontend/technology.md':
        print(doc)
        doc_with_error = doc

embeddings = OllamaEmbeddings(model=MODEL) # type: ignore

embedding_model_name = "thenlper/gte-base"

embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"device": "cpu", "batch_size": 100})

db = Chroma("vectorstores", embedding_model, persist_directory="vectorstores")

# Process: split & embeddings
def split_and_embedding(doc):
    print("Processing:", doc.metadata.get("source"))
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    if doc.page_content is None or doc.page_content == '':
        return

    md_header_splits = markdown_splitter.split_text(doc.page_content)

    chunk_size = 500
    chunk_overlap = 0
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    splits = text_splitter.split_documents(md_header_splits)

    db.add_texts(list(map(lambda split: split.page_content, splits)))


# split_and_embedding(doc_with_error)

for doc in docs:
    split_and_embedding(doc)

# Persist vectors
db.persist()

# Output
llm = Ollama(model=MODEL,
             verbose=True,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

question = "how install sneakers?"
result = qa_chain({"query": question})
print("Answer:", result)
