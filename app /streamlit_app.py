import os 
import tempfile 
import streamlit as st 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.title("Climate Research Assistant")
st.write("Upload a climate or air quality-related PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

question = st.text_input("Ask a question")

if uploaded_file and question:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=GOOGLE_API_KEY,
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GOOGLE_API_KEY,
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a climate research assistant.

        Use ONLY the provided context to answer the question.
        If the answer is not in the context, say you do not know.

        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = prompt | llm 

    response = chain.invoke({
        "context": context,
        "question": question,
    })

    def extract_response_text(response):
        if isinstance(response.content, str):
            return response.content

        if isinstance(response.content, list):
            return "\n".join(
                item.get("text", "")
                for item in response.content
                if item.get("type") == "text"
            )

        return str(response.content)


    st.subheader("Answer")
    answer_text = extract_response_text(response)
    st.markdown(answer_text)

    st.subheader("Sources")
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "Unknown")
        st.write(f"{i}. Page {page + 1}")