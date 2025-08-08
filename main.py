import os
import tempfile
import requests
from uuid import uuid4
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(
    title="LLM Query Retrieval (PDF/URL)",
    version="1.1",
    description="HackRx-compatible backend with PDF/XLSX query support"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In prod: restrict to allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store: doc_id -> FAISS vectorstore
DOCUMENTS = {}


def process_document_to_vectorstore(file_path: str) -> FAISS:
    """
    Load a PDF/XLSX file and build a FAISS vectorstore from chunks.
    """
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith((".xlsx", ".xls")):
        loader = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError("Unsupported file type (only PDF/XLSX/XLS supported)")

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def build_llm_chain(vectorstore: FAISS):
    """
    Build a ConversationalRetrievalChain with Groq LLM and the provided retriever.
    """
    try:
        llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
    except TypeError:
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=False
    )


class RunRequest(BaseModel):
    document_id: Optional[str] = None
    document_url: Optional[str] = None
    questions: List[str]


class RunResponse(BaseModel):
    answers: List[str]


@app.get("/")
async def root():
    return {"status": "ok", "message": "HackRx backend running"}


@app.post("/upload")
async def upload_single_file(file: UploadFile = File(...)):
    try:
        tmp_dir = tempfile.mkdtemp()
        save_path = os.path.join(tmp_dir, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())

        vectorstore = process_document_to_vectorstore(save_path)
        doc_id = str(uuid4())
        DOCUMENTS[doc_id] = vectorstore

        return {"message": "processed", "document_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Single upload failed: {str(e)}")


@app.post("/upload-multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    try:
        merged_vectorstore = None

        for file in files:
            tmp_dir = tempfile.mkdtemp()
            save_path = os.path.join(tmp_dir, file.filename)
            with open(save_path, "wb") as f:
                f.write(await file.read())

            vectorstore = process_document_to_vectorstore(save_path)

            if merged_vectorstore is None:
                merged_vectorstore = vectorstore
            else:
                merged_vectorstore.merge_from(vectorstore)

        if merged_vectorstore is None:
            raise HTTPException(status_code=400, detail="No valid files uploaded")

        doc_id = str(uuid4())
        DOCUMENTS[doc_id] = merged_vectorstore

        return {"message": "processed", "document_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multiple upload failed: {str(e)}")


@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def hackrx_run(req: RunRequest):
    try:
        if not req.document_id and not req.document_url:
            raise HTTPException(status_code=400, detail="You must provide either document_id or document_url")

        if req.document_id:
            if req.document_id not in DOCUMENTS:
                raise HTTPException(status_code=404, detail="document_id not found; please upload first")
            vectorstore = DOCUMENTS[req.document_id]
        else:
            if not req.document_url.startswith("http"):
                raise HTTPException(status_code=400, detail="Invalid document_url")

            resp = requests.get(req.document_url, timeout=30)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download URL, status {resp.status_code}")

            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, "downloaded.pdf")
            with open(tmp_path, "wb") as f:
                f.write(resp.content)

            vectorstore = process_document_to_vectorstore(tmp_path)

        qa_chain = build_llm_chain(vectorstore)

        answers = []
        for q in req.questions:
            if not q.strip():
                answers.append("")
                continue
            try:
                out = qa_chain({"question": q})
                ans = out.get("answer") or out.get("result") or out.get("text") or "" if isinstance(out, dict) else str(out)
                answers.append(ans)
            except Exception as e:
                answers.append(f"Error generating answer: {str(e)}")

        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
