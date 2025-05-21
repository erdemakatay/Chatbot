Bu projede, Türkçe cümlelerin öznel (duygusal veya yorumsal) mi yoksa nesnel (tarafsız ve gerçeklere 
dayalı) mı olduğunu otomatik olarak belirleyebilen bir chatbot sistemi geliştirilmiştir.  

Projenin temel amacı, kullanıcıdan alınan bir cümleyi analiz ederek bu cümlenin nesnel mi yoksa öznel 
mi olduğunu belirleyen ve bu sınıflandırmayı anlaşılır bir şekilde geri döndüren bir sistem 
oluşturmaktır. Bu amaç doğrultusunda; metin verileri üzerinde ön işleme, makine öğrenmesi ile model 
eğitimi ve kullanıcı etkileşimi sağlayan grafiksel arayüz adımları bir araya getirilmiştir. 

Proje iki ana bileşenden oluşmaktadır:  
1. Model Eğitimi (training.py): Öznel ve nesnel cümlelerden oluşan bir veri kümesi üzerinde bir Lojistik 
Regresyon modeli eğitilir. Bu aşama, veri ön işleme, özellik çıkarma (TF-IDF) ve model değerlendirme 
adımlarını içerir.  
2. Web Uygulaması (app.py): Eğitilen model ve özellik çıkarıcı (TF-IDF vektörleyici) kullanılarak, 
kullanıcıların cümlelerini sınıflandırabilecekleri interaktif bir web arayüzü (chatbot) Streamlit ile 
oluşturulur.


 Geleneksel Makine öğrenmesi ile olan chatbot Projesi Nasıl çalıştırılır?  (app.py)

 Terminalde: python -m streamlit run app.py  -> çalıştır

 ve streamlit arayüzü gelecek onla konuşmaya başlayacaksın. 
