import requests
import warnings
import io
from typing import List
from PyPDF2 import PdfReader
from phi.agent import Agent
from phi.embedder.ollama import OllamaEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.model.ollama import Ollama
from phi.document.reader.pdf import PDFUrlReader
from phi.document import Document

# Suppress SSL warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Create custom PDF reader
class CustomPDFReader(PDFUrlReader):
    def read(self, url: str) -> List[Document]:
        """Read PDF from URL without SSL verification"""
        response = requests.get(url, verify=False)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch PDF from {url}, status code: {response.status_code}")
        
        return self.read_bytes(response.content, source=url)
    
    def read_bytes(self, content: bytes, source: str = None) -> List[Document]:
        """Process PDF content and return documents"""
        pdf_file = io.BytesIO(content)
        pdf_reader = PdfReader(pdf_file)
        documents = []

        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():  # Only create document if there's text content
                doc = Document(
                    content=text,  # Changed from text to content
                    source=source,
                    metadata={"page": page_num + 1}
                )
                documents.append(doc)

        return documents

# Create knowledge base with custom reader
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://tuba.gov.tr/files/yayinlar/bilim-ve-dusun/TUBA-978-605-2249-48-2_Ch9.pdf"],
    vector_db=LanceDb(
        table_name="Ai",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=OllamaEmbedder(model="llama3.1:8b"),
    ),
    reader=CustomPDFReader(),
)

# Load the knowledge base
try:
    knowledge_base.load()
except Exception as e:
    print(f"Error loading knowledge base: {str(e)}")
    raise

# Initialize the agent
agent = Agent(
    model=Ollama(id="llama3.1:8b"),
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True,
)



# Query the agent
agent.print_response(
    input("sorunuzu giriniz:"),
    stream=True,
)
