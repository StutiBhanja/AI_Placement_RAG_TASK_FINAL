"""V1 of the project document: LangChain basics (no retrieval, no vector database).

    Input -> Prompt -> LLM -> Output Parser -> Chain / Runnable -> Output

Run:  python -m src.v1_basics
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from . import config
from .llm import make_llm, safe_invoke


def build_chain(llm=None):
    # 1. Prompt template: a reusable text with {placeholders}
    prompt = ChatPromptTemplate.from_template(
        "You are a friendly placement-preparation mentor.\n"
        "Explain the following interview topic in 3 short sentences for a beginner.\n\n"
        "Topic: {topic}"
    )
    # 2. LLM            3. Output parser (message object -> plain text)
    llm = llm or make_llm()
    parser = StrOutputParser()
    # 4. Runnable: any step that can be piped with |  (here: tidy the text)
    tidy = RunnableLambda(lambda text: " ".join(text.split()))
    # 5. Chain = prompt | llm | parser | tidy
    return prompt | llm | parser | tidy


if __name__ == "__main__":
    config.get_api_key()
    chain = build_chain()
    print(safe_invoke(chain, {"topic": "What is the STAR method?"}))
