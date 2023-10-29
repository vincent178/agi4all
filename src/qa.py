from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain import hub
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

embedding_model_name = "thenlper/gte-base"

embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"device": "cpu", "batch_size": 100})

db = Chroma("vectorstores", embedding_model, persist_directory="vectorstores")

llm = Ollama(model="mistral:latest",
             verbose=True,
             temperature=0,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

question = "how to configure Docker for sneakers application?"
result = qa_chain({"query": question})
