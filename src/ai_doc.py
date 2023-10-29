from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Initialize
embedding_model = HuggingFaceEmbeddings(
                model_name="thenlper/gte-base",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"device": "cpu", "batch_size": 100})

db = Chroma("vectorstores", embedding_model, persist_directory="vectorstores")

# Load
loader = DirectoryLoader('../../GOAT/engineering-docs/website/docs', glob="**/*.md", use_multithreading=True)
docs = loader.load()

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


map(split_and_embedding, docs)

# Persist vectors
db.persist()

