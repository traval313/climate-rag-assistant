from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from config import GOOGLE_API_KEY

DB_Path="chroma_db"

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=GOOGLE_API_KEY,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

vectorstore = Chroma(
    persist_directory=DB_Path,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 8,
#         "fetch_k": 30,
#     },
# )


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

question = input("Ask a question about your PDF: ")
docs = retriever.invoke(question)

print("\nRetrieved docs:\n")
for i, doc in enumerate(docs, start=1):
    print(f"--- Doc {i} ---")
    print(doc.metadata)
    print(doc.page_content[:500])

context = "\n\n".join(doc.page_content for doc in docs)

chain = prompt | llm 

response = chain.invoke({
    "context": context,
    "question": question,
})

print("\nAnswer:\n")
print(response.text)

print("\nSources:\n")
for i, doc in enumerate(docs, start=1):
    source = doc.metadata.get("source", "Unknown source")
    page = doc.metadata.get("page", "Unknown page")
    print(f"{i}. {source}, page {page + 1}")