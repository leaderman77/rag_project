from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from pydantic import BaseModel
from typing import Union
from models import QuestionRequest, AnswerResponse
from rag_engine import RAGEngine

app = FastAPI()
rag_engine = RAGEngine()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GradioRequest(BaseModel):
    data: Union[list, dict]

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        answer = await rag_engine.ask_question(request.question, request.retriever_type)
        return AnswerResponse(answer=answer)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run/predict")
async def gradio_predict(request: GradioRequest):
    try:
        if isinstance(request.data, list):
            question, retriever_type = request.data
        elif isinstance(request.data, dict):
            question = request.data.get("question", "")
            retriever_type = request.data.get("retriever_type", "base")
        else:
            raise ValueError("Invalid data format")

        answer = await rag_engine.ask_question(question, retriever_type)
        return {"data": [answer]}
    except Exception as e:
        return {"error": str(e)}

# Gradio interface
def gradio_ask(question, retriever_type):
    response = rag_engine.ask_question(question, retriever_type)
    return response

iface = gr.Interface(
    fn=gradio_ask,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your question here...", label="Question"),
        gr.Radio(["base", "sentence_window", "auto_merging", "knowledge_graph"], label="Retriever Type")
    ],
    outputs="text",
    title="RAG Q&A System",
    description="Ask a question and select a retriever type to get an answer from the RAG system."
)

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)