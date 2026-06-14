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

st.set_page_config(
    page_title="Climate Research Assistant",
    page_icon="🌎",
    layout="wide",
)

st.title("🌎 Climate Research Assistant")
st.write("Upload a climate or air quality-related PDF to ask questions or generate a structured research summary.")

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

def save_uploaded_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name        

def load_pdf_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GOOGLE_API_KEY,
    )


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

tab1, tab2 = st.tabs(["Ask Questions", "Summarize Paper"])

with tab1: 
    st.subheader("Ask a Question About the Paper")

    question = st.text_input(
        "Question",
        placeholder="Example: How did climate change affect school infrastructure?",
    )


    if uploaded_file and question: 
        with st.spinner("Reading PDF and searching for relative context..."):
            temp_path = save_uploaded_pdf(uploaded_file)
            documents = load_pdf_documents(temp_path)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )

            chunks = splitter.split_documents(documents)

            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=GOOGLE_API_KEY,
            )

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
            )

            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 8,
                    "fetch_k": 40
                },
            )

            docs = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)
        
        with st.spinner("Generating answer..."): 
            llm = get_llm()

            prompt = ChatPromptTemplate.from_template(
                """
                You are a climate research assistant.

                Use ONLY the provided context to answer the question.
                If the answer is not in the context, say you do not know.

                Format your answer with:
                - clear bullet points when useful
                - bolded key terms
                - short paragraphs

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
        
        st.subheader("Answer")
        st.markdown(extract_response_text(response))


        st.subheader("Sources")
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page", "Unknown")
            page_display = page + 1 if isinstance(page, int) else page

            with st.expander(f"Source {i}: Page {page_display}"):
                st.write(doc.page_content[:1000])

with tab2: 
    st.subheader("Generate a Structured Research Paper Summary")

    st.write(
        "This creates a structured summary including the research question, methods, dataset, findings, limitations, and climate/air quality relevance."
    )

    summarize_button = st.button("Summarize Paper", type="primary")

    if uploaded_file and summarize_button:
        with st.spinner("Reading and summarizing paper..."):
            temp_path = save_uploaded_pdf(uploaded_file)
            documents = load_pdf_documents(temp_path)

            full_text = "\n\n".join(
                f"[Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
                for doc in documents
            )

            max_chars = 50000
            paper_text = full_text[:max_chars]

            llm = get_llm()

            summary_prompt = ChatPromptTemplate.from_template(
                """
                You are a climate and environmental research assistant.

                Read the research paper text below and create a structured summary.

                Use this exact format:

                # Research Paper Summary

                ## 1. Research Question
                Explain the main question or problem the paper investigates.

                ## 2. Methods
                Explain the study design, methodology, or analytical approach.

                ## 3. Dataset / Participants / Study Context
                Describe the dataset, participants, location, sample size, or study context.

                ## 4. Key Findings
                Summarize the most important findings in bullet points.

                ## 5. Limitations
                Identify limitations stated in the paper. If limitations are not clearly stated, say so.

                ## 6. Why This Matters for Climate / Air Quality
                Explain why this paper is important for climate, environment, public health, education, or air quality work.

                Rules:
                - Use only the provided paper text.
                - Do not invent details.
                - Use clear Markdown formatting.
                - Keep the summary concise but specific.

                Paper text:
                {paper_text}
                """
            )

            summary_chain = summary_prompt | llm

            summary_response = summary_chain.invoke({
                "paper_text": paper_text,
            })

        st.subheader("Research Paper Summary")
        st.markdown(extract_response_text(summary_response))

        st.caption(f"Pages loaded: {len(documents)}")
        st.caption(f"Characters summarized: {len(paper_text)}")

    elif summarize_button and not uploaded_file:
        st.warning("Please upload a PDF first.")