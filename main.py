import os
import time  #Zamanlayıcı kütüphanesi
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import google.generativeai as genai

API_KEY = "api_key"

# Google AI Ayarları,
# API Key sisteme tanıtma
genai.configure(api_key=API_KEY)

CATEGORIES = ["IADE_VE_DEGISIM", "KARGO_TESLIMAT", "URUN_KUSURU", "ODEME_FATURA", "ONERI_SIKAYET"]


def proje_baslat():
    print("Proje Başlatılıyor")

    # EXCEL dosyasını okuma
    # Dosya okunamaması durumunda sistemin çökmesini engellemek için try-except bloğu eklenmesi
    try:
        df = pd.read_excel("eticaret_nlp_dataset_1000.xlsx")
        print(f"Excel dosyası başarıyla okundu. Toplam {len(df)} satır veri var.")
    except Exception as e:
        print(f"HATA: Dosya okunamadı. Hata: {e}")
        return

    #Sütun isimlerini normalize edilmesi
    df.columns = df.columns.str.strip().str.lower()

    #Farklı isimlerin kodun anlayacağı ortak isimlere çevilmesi
    isim_degisikligi = {
        'konu': 'subject',
        'açıklama': 'body',
        'mesaj': 'body',
        'içerik': 'body',
        'ilgili departman': 'department',
        'i̇lgili departman': 'department',
        'department': 'department'
    }
    df.rename(columns=isim_degisikligi, inplace=True)

    # Konu ve metin başlıklarının birleştirilmesi.
    try:
        df['full_text'] = "Konu: " + df['subject'].astype(str) + " | İçerik: " + df['body'].astype(str)
    except KeyError:
        print("HATA: Sütun isimleri bulunamadı. Excel başlıkları: ", df.columns.tolist())
        return

    # Test verilerini oluşturma
    _, test_data = train_test_split(df, test_size=50, random_state=42)

    print(f"Test edilecek veri sayısı: {len(test_data)}")

    # Modelin ismini tanımlama
    MODEL_NAME = 'gemini-2.5-flash'
    print(f"Yapay Zeka Modeli: ({MODEL_NAME})")

    try:
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return

    print("\nSINIFLANDIRMA BAŞLIYOR \n")

    #Gerçek değerlerin listesini oluşturma
    y_true = test_data['department'].tolist()

    #Tahmin değerleri için boş liste tanımlama
    y_pred = []

    counter = 1

    # full_text verisininin text değişkenine atanması
    for index, row in test_data.iterrows():
        text = row['full_text']

        #FEW-SHOT PROMPT
        prompt = f"""
                Sen uzman bir e-ticaret sınıflandırma asistanısın.
                Görevin, müşteri mesajını analiz edip aşağıdaki 5 kategoriden en doğrusuna atamaktır.

                KATEGORİLER:
                ['IADE_VE_DEGISIM', 'KARGO_TESLIMAT', 'URUN_KUSURU', 'ODEME_FATURA', 'ONERI_SIKAYET']

                ### ÖNEMLİ DETAYLAR :
                1. Eğer müşteri "Para iadesi", "Kartıma geri ödeme", "Tutar yatmadı" diyorsa -> ODEME_FATURA seç.
                 (Çünkü bu finansal bir işlemdir).
                2. Eğer müşteri "Ürünü geri göndermek", "İade kodu", 
                "Değişim", "Beden uymadı" diyorsa -> IADE_VE_DEGISIM seç.
                3. Eğer ürün "Kırık", "Bozuk", "Lekeli", "Eksik" geldiyse -> URUN_KUSURU seç.

                ### ÖRNEKLER (FEW-SHOT LEARNING):
                Mesaj: "İade ettiğim ürünün parası 3 gündür hesabıma yatmadı."
                Cevap: ODEME_FATURA

                Mesaj: "Kazağın bedeni küçük geldi, nasıl geri gönderebilirim?"
                Cevap: IADE_VE_DEGISIM

                Mesaj: "Kutudan çıkan şarj aleti çalışmıyor, bozuk."
                Cevap: URUN_KUSURU

                Mesaj: "Kredi kartımdan iki kere çekim yapılmış."
                Cevap: ODEME_FATURA

                ### ŞİMDİ SENİN SIRAN:
                Mesaj: "{text}"

                Cevap (Sadece kategori ismi):
                """

        try:
            #Promptun yapay zekaya göndilmesi
            response = model.generate_content(prompt)

            #Yapay zekadan gelen cevabın temizlenmesi
            tahmin = response.text.strip()
            tahmin_temiz = "BELIRSIZ"
            for cat in CATEGORIES:
                if cat in tahmin:
                    tahmin_temiz = cat
                    break


            y_pred.append(tahmin_temiz)
            print(f"[{counter}/50] Gerçek: {row['department']} -> Tahmin: {tahmin_temiz}")

        except Exception as e:
            print(f"Hata oluştu (Muhtemelen kota): {e}")
            y_pred.append("HATA")

        counter += 1

        #Sorgu sınırına takılmamak için 4 saniye beklenmesi (429 hatası)
        time.sleep(4)

    #SONUÇ RAPORU
    print("\n" + "=" * 30)
    acc = accuracy_score(y_true, y_pred)
    print(f"Model Başarı Oranı: %{acc * 100:.2f}")
    print("=" * 30)

    print("\nDetaylı Rapor:")
    print(classification_report(y_true, y_pred, zero_division=0))

    #Confusion matrix çizdirilmesi
    print("Grafik çiziliyor...")
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CATEGORIES, yticklabels=CATEGORIES, cmap='Blues')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Olan')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    proje_baslat()