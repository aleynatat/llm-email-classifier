import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import google.generativeai as genai
import time

# --- AYARLAR ---
# BURAYA KENDİ API KEY'İNİ YAPIŞTIR
API_KEY = "AIzaSyCD2hCA69I6fGcMOitMVU2wTPyNAnu_IBg"

# Google AI Ayarları
genai.configure(api_key=API_KEY)

CATEGORIES = ["IADE_VE_DEGISIM", "KARGO_TESLIMAT", "URUN_KUSURU", "ODEME_FATURA", "ONERI_SIKAYET"]

# Sayfa Ayarları (Sekme adı ve ikonu)
st.set_page_config(page_title="AI E-Posta Asistanı", page_icon="📧", layout="wide")


# --- FONKSİYONLAR ---

@st.cache_resource  # Modeli önbelleğe alır, her seferinde yüklemez (Hızlandırır)
def get_model():
    return genai.GenerativeModel('gemini-2.5-flash')


def classify_email(text, model):
    """Tek bir metni sınıflandırır"""
    prompt = f"""
    Sen uzman bir e-ticaret sınıflandırma asistanısın.
    Görevin, müşteri mesajını analiz edip aşağıdaki 5 kategoriden en doğrusuna atamaktır.

    KATEGORİLER: {CATEGORIES}

    KURALLAR:
    1. Para/Kart/Fatura -> ODEME_FATURA
    2. İade/Değişim/Beden -> IADE_VE_DEGISIM
    3. Kırık/Bozuk/Eksik -> URUN_KUSURU

    ÖRNEKLER:
    "Param yatmadı" -> ODEME_FATURA
    "Beden olmadı" -> IADE_VE_DEGISIM

    Mesaj: "{text}"
    Cevap (Sadece kategori ismi):
    """
    try:
        response = model.generate_content(prompt)
        tahmin = response.text.strip()
        # Temizlik
        for cat in CATEGORIES:
            if cat in tahmin:
                return cat
        return "BELIRSIZ"
    except Exception as e:
        return f"HATA: {e}"


# --- ARAYÜZ (UI) TASARIMI ---

# Yan Menü (Sidebar)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
st.sidebar.title("Kontrol Paneli")
page = st.sidebar.radio("Mod Seçiniz:", ["Canlı Test (Demo)", "Toplu Analiz & Rapor"])

st.sidebar.info("Bu proje Doğal Dil İşleme dersi için hazırlanmıştır.")

# --- SAYFA 1: CANLI TEST (DEMO) ---
if page == "Canlı Test (Demo)":
    st.title("📧 AI E-Posta Sınıflandırma Sistemi")
    st.markdown("Aşağıya bir müşteri e-postası yapıştırın ve yapay zekanın hangi departmana yönlendireceğini görün.")

    # Kullanıcıdan Veri Alma
    user_input = st.text_area("Müşteri Mesajı:",
                              placeholder="Örn: Merhaba, kargom hala gelmedi, iptal etmek istiyorum...", height=150)

    if st.button("🚀 Analiz Et"):
        if user_input:
            with st.spinner('Yapay Zeka düşünüyor...'):
                model = get_model()
                kategori = classify_email(user_input, model)

            # Sonucu Göster
            st.success("Analiz Tamamlandı!")
            st.subheader(f"Yönlendirilen Departman:")

            # Renkli Kutu Tasarımı
            if kategori == "ODEME_FATURA":
                st.info(f"💳 {kategori}")
            elif kategori == "KARGO_TESLIMAT":
                st.warning(f"📦 {kategori}")
            elif kategori == "URUN_KUSURU":
                st.error(f"🛠️ {kategori}")
            elif kategori == "IADE_VE_DEGISIM":
                st.success(f"🔄 {kategori}")
            else:
                st.primary(f"📝 {kategori}")

        else:
            st.warning("Lütfen bir mesaj giriniz.")

# --- SAYFA 2: TOPLU ANALİZ VE RAPOR ---
elif page == "Toplu Analiz & Rapor":
    st.title("📊 Performans Raporu")
    st.markdown("Excel dosyasındaki 1000 veriden rastgele **50 tanesi** seçilip test edilecektir.")

    if st.button("Testi Başlat (Yaklaşık 2-3 dk sürer)"):
        status_text = st.empty()
        progress_bar = st.progress(0)

        try:
            # Veriyi Oku
            df = pd.read_excel("eticaret_nlp_dataset_1000.xlsx")

            # Ön İşleme
            df.columns = df.columns.str.strip().str.lower()
            rename_map = {'konu': 'subject', 'açıklama': 'body', 'içerik': 'body',
                          'ilgili departman': 'department', 'department': 'department'}
            df.rename(columns=rename_map, inplace=True)
            df['full_text'] = "Konu: " + df['subject'].astype(str) + " | İçerik: " + df['body'].astype(str)

            # Split
            _, test_data = train_test_split(df, test_size=50, random_state=42)

            model = get_model()
            y_true = test_data['department'].tolist()
            y_pred = []

            counter = 0
            for index, row in test_data.iterrows():
                # İlerleme Çubuğunu Güncelle
                counter += 1
                status_text.text(f"Analiz ediliyor: {counter}/50 - {row['department']}")
                progress_bar.progress(counter / 50)

                # Tahmin
                pred = classify_email(row['full_text'], model)
                y_pred.append(pred)

                # API Kotası için bekleme
                time.sleep(4)

            status_text.text("✅ Test Tamamlandı!")

            # --- METRİKLERİ GÖSTER ---
            acc = accuracy_score(y_true, y_pred)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Toplam Test Verisi", value="50 Adet")
            with col2:
                st.metric(label="Doğruluk Oranı (Accuracy)", value=f"%{acc * 100:.2f}")

            # --- GRAFİK ---
            st.subheader("Confusion Matrix (Karmaşıklık Matrisi)")
            fig, ax = plt.subplots(figsize=(10, 6))
            cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=CATEGORIES, yticklabels=CATEGORIES, cmap='Blues', ax=ax)
            plt.ylabel('Gerçek')
            plt.xlabel('Tahmin')
            st.pyplot(fig)

            # --- HATALI TAHMİNLER TABLOSU ---
            st.subheader("Hatalı Tahminler (İnceleme)")
            errors = []
            for i in range(len(y_true)):
                if y_true[i] != y_pred[i]:
                    errors.append([y_true[i], y_pred[i]])

            if errors:
                error_df = pd.DataFrame(errors, columns=["Gerçek", "Tahmin"])
                st.table(error_df)
            else:
                st.success("Tebrikler! Hata bulunamadı.")

        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")


