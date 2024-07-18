from fastapi import FastAPI
from typing_extensions import TypedDict

from langchain_core.runnables import RunnableLambda
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS

from langgraph.graph import END, StateGraph, START

from langserve import add_routes


def inp(question: str) -> dict:
    return {"question": question}


def out(state: dict):
    result = state[1]["generate"]["generation"]  # 0: retrieve, 1: generate
    return result


def format_docs(docs):
    """Convert retrieved documents into str format"""

    return "\n\n".join(doc.page_content for doc in docs)


def setup_pdf_retriever(pdf_file_path: str):
    """Read a PDF file and setup an ensemble retriever"""

    pages = PyMuPDFLoader(file_path=pdf_file_path).load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=100
    )
    docs = text_splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-nli",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore_faiss = FAISS.from_documents(docs, embeddings)
    faiss_retriever = vectorstore_faiss.as_retriever(
        search_kwargs={"k": 1}
    )  # dense search
    bm25_retriever = BM25Retriever.from_documents(docs)  # sparse search
    bm25_retriever.k = 1

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],  # sum equals to 1
    )
    return retriever


# Graph
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """

    question: str
    generation: str
    documents: str


def retrieve(state):
    """Define retrieval"""

    question = state["question"]
    documents = retriever.invoke(question)
    state["documents"] = documents
    return state


def generate(state):
    """Define generation"""

    question = state["question"]
    documents = state["documents"]
    documents = format_docs(documents)
    generation = rag_chain.invoke({"context": documents, "question": question})
    state["generation"] = generation
    return state


# Agentic workflow building
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("generate", generate)  # generate
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
compiled_workflow = workflow.compile()


# App building
local_llm = "Llama-3-Open-Ko-8B-Instruct-preview-Q5_K_M_assistant_wo_system.gguf"
basic_doc_path = ""  # YOUR PDF FILENAME HERE
retriever = setup_pdf_retriever(basic_doc_path)
prompt = PromptTemplate(  # Instruct in Enlgish, but force to generate in Korean
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering strictly based on context.
    Do not guess the answer. You must only use Context, not other information for answer. Only incude highly related information in Context. 
    Just say you don't know if you can't find related information in Context. 
    Answer only in Korean. <|eot_id|><|start_header_id|>user<|end_header_id|>
    \n Question: {question}
    \n Context: {context}
    \n Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["context", "question"],
)

llm = ChatOllama(model=local_llm, temperature=0)
rag_chain = prompt | llm | StrOutputParser()
final_chain = RunnableLambda(inp) | compiled_workflow | RunnableLambda(out)

fastapi_app = FastAPI(  # Initialize FastAPI app
    title="PDF Q&A bot",
    version="0.1",
    description="PDF Q&A Chatbot Server with Langgraph + Langserve",
)
add_routes(fastapi_app, final_chain)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(fastapi_app, host="0.0.0.0", port=5000)
