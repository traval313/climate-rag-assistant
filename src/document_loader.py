from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "data/sample_pdfs/es-paper2.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(documents)

print(f"Loaded {len(documents)} pages")
print(f"Created {len(chunks)} chunks")
print(chunks[0].page_content[:500])
print(chunks[0].metadata)
