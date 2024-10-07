from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str
    retriever_type: str


class AnswerResponse(BaseModel):
    answer: str