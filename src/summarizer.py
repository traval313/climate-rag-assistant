import sys 
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config import GOOGLE_API_KEY

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


def load_pdf_text(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    full_text = "\n\n".join(
        f"[Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
        for doc in documents
    )

    return full_text, documents


def summarize_paper(pdf_path):
    full_text, documents = load_pdf_text(pdf_path)

    # Keeps prompt from getting too huge while you're using free Gemini quota.
    max_chars = 50000
    paper_text = full_text[:max_chars]

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GOOGLE_API_KEY,
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a climate and environmental research assistant.

        Read the research paper text below and create a structured summary.

        Use this exact format:

        # Research Paper Summary

        ## 1. Research Question
        Explain the main question or problem the paper investigates in just 1-2 lines.

        ## 2. Methods
        Explain the study design, methodology, or analytical approach in 2-3 sentences.

        ## 3. Dataset / Participants / Study Context
        Describe the dataset, participants, location, sample size, or study context concisely.

        ## 4. Key Findings
        Summarize the most important findings in bullet points.

        ## 5. Limitations
        Identify limitations stated in the paper. If limitations are not clearly stated, say so.

        ## 6. Why This Matters for Climate / Air Quality
        Explain why this paper is important for climate, environment, public health, education, or air quality work in 1-2 lines.

        Rules:
        - Use only the provided paper text.
        - Do not invent details.
        - Use clear Markdown formatting.
        - Keep the summary concise but specific.

        Paper text:
        {paper_text}
        """
    )

    chain = prompt | llm

    response = chain.invoke({
        "paper_text": paper_text,
    })

    summary = extract_response_text(response)

    print(summary)
    print("\n---")
    print(f"Pages loaded: {len(documents)}")
    print(f"Characters summarized: {len(paper_text)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/summarizer.py path/to/paper.pdf")
        sys.exit(1)

    summarize_paper(sys.argv[1])