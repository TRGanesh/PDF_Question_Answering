import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import StringIO

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session_state to store persistent data
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Append question and answer to the chat history in session_state
    st.session_state.chat_history.append({"question": user_question, "answer": response["output_text"]})

    # Display the entire chat history with download button
    chat_history_text = "\n\n".join(
        [f"Q: {entry['question']}\nA: {entry['answer']}\n----" for entry in reversed(st.session_state.chat_history)]
    )
    
    # Add a download button for text file
    download_text_button = st.download_button("Download Chat History (Text)", chat_history_text, key="text_button")

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    # Add "Get Answer" button next to the input box
    if st.button("Get Answer"):
        if user_question:
            user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Style the chat history text area
    st.markdown(
        """
        <style>
            .chat-history {
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 10px;
                overflow-y: scroll;
                max-height: 400px;
            }
            .message {
                margin-bottom: 10px;
            }
            .question {
                color: #007BFF;
                font-weight: bold;
            }
            .answer {
                color: #28A745;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Display the entire chat history with the specified styles
    st.markdown(
        "<div class='chat-history'>{}</div>".format(
            "\n".join(
                [
                    f"<div class='message question'>Q: {entry['question']}</div><div class='message answer'>A: {entry['answer']}</div>"
                    for entry in reversed(getattr(st.session_state, 'chat_history', []))
                ]
            )
        ),
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
