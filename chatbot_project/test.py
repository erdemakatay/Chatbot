import langchain
import openai
import pandas
import faiss
import tiktoken
import dotenv
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter # Metinleri bölmek için
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document # Document nesnesini import ediyoruz

print("Kütüphaneler başarıyla import edildi!")

# .env dosyasındaki değişkenleri yükle (OPENAI_API_KEY)
load_dotenv()

# OpenAI API anahtarının yüklenip yüklenmediğini kontrol et
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("HATA: OPENAI_API_KEY ortam değişkeni bulunamadı. Lütfen .env dosyanızı kontrol edin.")
    exit()

# Veri Kümesini Yükleme ve Hazırlama
DATA_PATH = "ÖZNEL-NESNEL VERİ KÜMESİ.csv"

def load_and_prepare_data(file_path: str) -> list[Document]:
    """CSV dosyasını yükler ve Langchain Document nesnelerine dönüştürür."""
    try:
        # Sadece ilk iki sütunu oku ve onlara isim ver
        # CSV'nizde ilk satır başlık olduğu için header=0 doğru,
        # names parametresi verdiğimiz için bu isimler kullanılacak.
        df = pd.read_csv(file_path, usecols=[0, 1], names=['Cümle', 'Tür_Etiketi'], header=0, encoding='utf-8')
        # İkinci "Tür" sütununu "Tür_Etiketi" olarak aldık ki "deger" ile karışmasın ve kontrol edelim.
    except FileNotFoundError:
        print(f"HATA: Veri dosyası bulunamadı: {file_path}")
        exit()
    except Exception as e:
        print(f"HATA: Veri dosyası okunurken bir hata oluştu: {e}")
        exit()

    # Şimdi 'Cümle' ve 'Tür_Etiketi' sütunlarını kontrol ediyoruz.
    if "Cümle" not in df.columns or "Tür_Etiketi" not in df.columns:
        print(f"HATA: CSV dosyasında beklenen 'Cümle' ve 'Tür_Etiketi' sütunları okunamadı. Okunan sütunlar: {df.columns.tolist()}")
        exit()

    documents = []
    for _, row in df.iterrows():
        cumle = str(row["Cümle"])
        # 'Tür_Etiketi' sütununu etiket olarak kullanıyoruz.
        deger = str(row["Tür_Etiketi"])

        # Cümle veya değer boşsa veya sadece boşluk içeriyorsa atla
        if not cumle.strip() or not deger.strip():
            # print(f"Atlanan satır (boş içerik): Cümle='{cumle}', Tür='{deger}'") # Hata ayıklama için
            continue
        
        # "Öznel" veya "Nesnel" dışında bir değer varsa atla (isteğe bağlı, veri temizliği için)
        if deger.lower() not in ["öznel", "nesnel"]:
            # print(f"Atlanan satır (geçersiz etiket): Cümle='{cumle}', Tür='{deger}'") # Hata ayıklama için
            continue

        documents.append(Document(page_content=cumle, metadata={"label": deger}))
    
    if not documents:
        print("HATA: CSV dosyasından hiçbir geçerli doküman yüklenemedi. Lütfen dosya formatını ve içeriğini kontrol edin.")
        exit()

    print(f"{len(documents)} adet doküman başarıyla yüklendi.")
    return documents

# Gömme (Embeddings) ve Vektör Veritabanı Oluşturma
def create_vector_store(documents: list[Document]):
    """Belgelerden bir FAISS vektör veritabanı oluşturur."""
    print("Gömme modeli yükleniyor...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # Önerilen ve uygun maliyetli bir embedding modeli
    print("Vektör veritabanı oluşturuluyor... Bu işlem biraz zaman alabilir.")
    try:
        vector_store = FAISS.from_documents(documents, embeddings)
        print("Vektör veritabanı başarıyla oluşturuldu.")
        return vector_store
    except Exception as e:
        print(f"HATA: Vektör veritabanı oluşturulurken bir hata oluştu: {e}")
        exit()

# RAG Zinciri Oluşturma
def create_rag_chain(vector_store):
    """Bir RAG zinciri oluşturur."""
    print("LLM (gpt-4o-mini) yükleniyor...")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0) # Kararlı cevaplar için temperature=0

    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # En alakalı 3 dokümanı getir

    # LLM'e verilecek talimat (prompt) şablonu
    # Bu şablon, LLM'e görevi ve nasıl cevap vermesi gerektiğini açıklar.
    # "Context" bölümüne, retriever tarafından bulunan benzer cümleler ve etiketleri gelecek.
    # "Question" bölümüne ise kullanıcının girdiği cümle gelecek.
    prompt_template_str = """
    Sen cümleleri 'Öznel' veya 'Nesnel' olarak sınıflandırmada uzman bir asistansın.
    Aşağıdaki veri kümesinden alınmış örneklere dayanarak, verilen 'Kullanıcı Cümlesi'nin 'Öznel' mi yoksa 'Nesnel' mi olduğunu belirle.
    Sadece 'Öznel' veya 'Nesnel' olarak cevap ver. Başka bir açıklama yapma.

    Veri Kümesinden Örnekler (Context):
    {context}

    Kullanıcı Cümlesi: {question}
    Sınıflandırma:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template_str, input_variables=["context", "question"]
    )

    # RetrievalQA zinciri, retriever'dan aldığı bilgiyi LLM'e prompt ile birlikte gönderir.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # En yaygın kullanılan ve tüm bulunan dokümanları contexte sığdırmaya çalışan tip
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False # Kaynak dokümanları yanıtta döndürme
    )
    print("RAG zinciri başarıyla oluşturuldu.")
    return qa_chain

def main():
    print("Chatbot başlatılıyor...")
    documents = load_and_prepare_data(DATA_PATH)
    if not documents:
        return

    vector_store = create_vector_store(documents)
    if not vector_store:
        return

    rag_chain = create_rag_chain(vector_store)
    if not rag_chain:
        return

    print("\n-----------------------------------------------------")
    print("Öznel/Nesnel Chatbot'a hoş geldiniz!")
    print("Bir cümle girin veya çıkmak için 'çıkış' yazın.")
    print("-----------------------------------------------------")

    while True:
        user_input = input("Siz: ")
        if user_input.lower() == "çıkış":
            print("Chatbot kapatılıyor. Hoşça kalın!")
            break
        if not user_input.strip():
            print("Lütfen bir cümle girin.")
            continue

        try:
            # Zinciri çalıştırıp sonucu alıyoruz
            # Langchain'in yeni sürümlerinde .invoke({"query": user_input}) kullanılır
            # Eski sürümlerde .run(user_input) veya .__call__(user_input) olabilir
            response = rag_chain.invoke({"query": user_input}) 
            # response['result'] içinde LLM'in cevabı bulunur
            classification = response.get('result', "Bir hata oluştu, cevap alınamadı.").strip()
            print(f"Bot: {classification}")
        except Exception as e:
            print(f"HATA: Cevap alınırken bir sorun oluştu: {e}")

if __name__ == "__main__":
    main()
