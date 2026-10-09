"""AI Placement Preparation Assistant using RAG.

Project Title        : AI Placement Preparation Assistant using RAG
Problem Statement    : Placement study material is spread across many PDFs (technical interview
                       notes, aptitude, HR questions, resume guidelines). Searching them by hand
                       is slow and inefficient.
Proposed Solution    : A RAG assistant (LangChain + Google Gemini + FAISS/ChromaDB + Streamlit)
                       that answers questions ONLY from the uploaded PDFs and shows the source
                       file and page. If the answer is not in the PDFs it says so.
Target Users         : Students preparing for campus placements.
Data / Documents     : data/technical_interview_.pdf, data/resume_guidelines.pdf,
                       data/hr_questions.pdf, data/aptitude.pdf
Expected Output      : A natural-language answer + source file and page number, or
                       "I could not find the answer in the uploaded documents."
"""
