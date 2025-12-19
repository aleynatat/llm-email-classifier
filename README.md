# LLM Tabanlı Akıllı E-Posta Sınıflandırma Sistemi

Bu proje, gelen e-postaların içeriklerini Doğal Dil İşleme (NLP) ve Büyük Dil Modelleri (LLM) kullanarak analiz eden ve ilgili departmanlara (Teknik Destek, Muhasebe, İK, Satış vb.) otomatik olarak sınıflandıran bir yapay zeka uygulamasıdır.

## Proje Hakkında

Kurumsal firmalara gelen yoğun e-posta trafiğini manuel olarak yönetmek zaman alıcı ve hata payı yüksek bir süreçtir. Bu sistem:
- Gelen metnin niyetini anlar.
- Karmaşık cümle yapılarını çözümler.
- E-postayı en doğru departmana yönlendirerek iş akışını hızlandırır.

## Temel Özellikler

- **NLP Odaklı Sınıflandırma:** Geleneksel anahtar kelime eşleştirmesi yerine, metnin semantik (anlamsal) analizini yapar.
- **LLM Entegrasyonu:** Google Gemini (veya OpenAI) API'leri kullanılarak yüksek doğrulukta sonuçlar üretir.
- **Sentetik Veri Üretimi:** Proje kapsamında oluşturulan özel bir veri seti ile test edilmiştir.
- **Modern Dashboard:** Streamlit kullanılarak hazırlanan analiz arayüzü ile sonuçları görselleştirir.

##  Kullanılan Teknolojiler

- **Dil:** Python 3.9+
- **Yapay Zeka:** LangChain, Google Generative AI (Gemini 1.5 Flash)
- **Arayüz:** Streamlit
- **Veri Yönetimi:** Pandas, JSON

## Kurulum

1. Bu depoyu klonlayın:
   ```bash
   git clone [https://github.com/kullanici-adin/llm-email-classifier.git](https://github.com/kullanici-adin/llm-email-classifier.git)
   cd llm-email-classifier