Bu Python kodu, bir PDF dosyasını uzaktan okur, metni çıkarır ve bu metni bir bilgi tabanı (knowledge base) içinde depolayarak, bir yapay zeka ajanının bu veriyi kullanarak sorulara cevap vermesini sağlar. Kodun her bir bölümünü detaylı bir şekilde açıklayayım:


## PDF Agentic RAG Görseli

Bu proje yapısı aşağıdaki görselde gösterilmiştir:

![PDF Agentic RAG](https://github.com/Therayz1/A-Agents/raw/main/PDF_AGENTIC_RAG.png)

### 1. **Gerekli Kütüphanelerin İçe Aktarılması**
```python
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
```

- `requests`: HTTP istekleri yapmak için kullanılır. Bu kütüphane ile PDF dosyasına erişim sağlanacaktır.
- `warnings`: Uyarıları engellemek için kullanılır. SSL uyarılarını görmezden gelmek için bu kullanılır.
- `io`: Byte veri akışlarını yönetir. PDF içeriği byte formatında okunacağı için kullanılır.
- `PyPDF2`: PDF dosyasındaki metni çıkarmak için kullanılan bir kütüphane.
- `phi.agent`, `phi.model`, `phi.vectordb`: Bu kütüphaneler, doğal dil işleme (NLP) ve yapay zeka işlevleri için kullanılır. Bu ajanlar, PDF içeriğinden sorulara cevap verebilen bir yapay zeka oluşturur.

### 2. **SSL Uyarılarının Engellenmesi**
```python
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
```
Bu satır, SSL doğrulaması yapılmamış HTTP isteklerinin uyarılarını engeller. Bu, güvenli olmayan bağlantılarda hata almanızı önler.

### 3. **Özel PDF Okuyucu Sınıfının Oluşturulması**
```python
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
```

- **`CustomPDFReader`**: Bu, PDF dosyasını URL üzerinden okuyacak özel bir sınıftır. 
    - `read` fonksiyonu, verilen URL'den PDF dosyasını indirir. SSL doğrulaması yapılmaz (`verify=False`).
    - `read_bytes` fonksiyonu, PDF dosyasını okur ve her sayfayı bir `Document` nesnesine dönüştürür. Bu `Document`, sayfa numarasını ve metni içerir.

### 4. **Bilgi Tabanı (Knowledge Base) Oluşturulması**
```python
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
```

- **`knowledge_base`**: Bu, PDF dosyasındaki veriyi bir veritabanına (vector database) kaydedecek bilgi tabanıdır. 
    - PDF dosyasının URL'si `urls` parametresinde verilmiştir.
    - `LanceDb` sınıfı, bu verilerin vektörler olarak kaydedilmesi ve arama yapılabilmesi için kullanılır.
    - `OllamaEmbedder`, PDF metnini vektöre dönüştürerek veritabanında depolar.
    - `CustomPDFReader`, metni çıkarmak için kullanılan özel PDF okuyucusudur.

### 5. **Bilgi Tabanının Yüklenmesi**
```python
try:
    knowledge_base.load()
except Exception as e:
    print(f"Error loading knowledge base: {str(e)}")
    raise
```

Bu kısım, oluşturduğumuz bilgi tabanını yüklemeye çalışır. Eğer bir hata oluşursa, hata mesajını yazdırır.

### 6. **Yapay Zeka Ajanının Oluşturulması**
```python
agent = Agent(
    model=Ollama(id="llama3.1:8b"),
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True,
)
```

- **`Agent`**: Bu, bir yapay zeka ajanıdır. 
    - `model` parametresine bir AI modelini (`Ollama` modelini) veriyoruz.
    - `knowledge` parametresine daha önce oluşturduğumuz `knowledge_base`'i veriyoruz. 
    - Bu ajan, PDF dosyasından aldığı verilerle sorulara cevap verebilecektir.
    - `show_tool_calls` ve `markdown` parametreleri, ajanın işleyişi hakkında detaylı bilgi vermek için ayarlanmıştır.

### 7. **Ajanın Soruya Cevap Vermesi**
```python
agent.print_response(
    input("sorunuzu giriniz:"),
    stream=True,
)
```

- **`agent.print_response`**: Bu, kullanıcının sorduğu soruyu alıp ajanın bu soruya vereceği cevabı yazdırır. 
    - Kullanıcıdan bir giriş alınır (`input` fonksiyonu ile).
    - Ajan, soruyu bilgi tabanındaki verilerle değerlendirir ve cevabını çıktı olarak verir.
    - `stream=True` ile cevap anlık olarak iletilir.

---

### Özet:
Bu kod, bir PDF dosyasından metin çıkartarak bu veriyi bir yapay zeka ajanına yükler. Kullanıcı, bu veriyi sorgulayabilir ve ajan, PDF içeriği üzerinden bilgi sağlayarak soruları yanıtlar. Kodu anlamanızı kolaylaştırmak için adım adım açıklamalarla detaylandırdım.
