import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Kullanacagimiz sutunlar
KATEGORIK = ["konum", "marka", "vites_tipi", "yakit_tipi", "kasa_tipi", "cekis"]
SAYISAL   = ["yil", "kilometre", "motor_hacmi", "motor_gucu"]

# Veri okunur ve model uygulama acilirken bir kez egitilir (cache).
@st.cache_resource
def veri_ve_model():
    df = pd.read_csv("cars1.csv")
    df = df[KATEGORIK + SAYISAL + ["fiyat"]].dropna()
    # Asiri uc fiyatlari (hatali ilanlar) at, model saglikli olsun
    df = df[(df["fiyat"] > 50000) & (df["fiyat"] < 15000000)]
    X = df[KATEGORIK + SAYISAL]
    y = df["fiyat"]
    onisleme = ColumnTransformer(
        [("kat", OneHotEncoder(handle_unknown="ignore"), KATEGORIK)],
        remainder="passthrough"
    )
    model = Pipeline([
        ("onisleme", onisleme),
        ("model", RandomForestRegressor(n_estimators=50, max_depth=15,
                                        random_state=42, n_jobs=-1))
    ])
    model.fit(X, y)
    return df, model

df, model = veri_ve_model()

st.title("🚗 Araba Fiyat Tahmini")
st.write("Aracın özelliklerini seç, tahmini fiyatı gör. (Türkiye 2. el araç verisi ile eğitilmiştir.)")

# Kategorik secimler icin menuleri veriden otomatik doldur
def secenekler(sutun):
    return sorted(df[sutun].dropna().unique().tolist())

col1, col2 = st.columns(2)
with col1:
    konum      = st.selectbox("Şehir", secenekler("konum"))
    marka      = st.selectbox("Marka", secenekler("marka"))
    yakit_tipi = st.selectbox("Yakıt tipi", secenekler("yakit_tipi"))
    vites_tipi = st.selectbox("Vites tipi", secenekler("vites_tipi"))
with col2:
    kasa_tipi  = st.selectbox("Kasa tipi", secenekler("kasa_tipi"))
    cekis      = st.selectbox("Çekiş", secenekler("cekis"))
    yil        = st.number_input("Model yılı", min_value=1990, max_value=2026, value=2018)
    kilometre  = st.number_input("Kilometre", min_value=0, value=100000, step=5000)

col3, col4 = st.columns(2)
with col3:
    motor_hacmi = st.number_input("Motor hacmi (cc)", min_value=800, value=1600, step=100)
with col4:
    motor_gucu  = st.number_input("Motor gücü (bg)", min_value=40, value=120, step=10)

if st.button("Fiyatı Tahmin Et", type="primary"):
    arac = pd.DataFrame([{
        "konum": konum, "marka": marka, "vites_tipi": vites_tipi,
        "yakit_tipi": yakit_tipi, "kasa_tipi": kasa_tipi, "cekis": cekis,
        "yil": yil, "kilometre": kilometre,
        "motor_hacmi": motor_hacmi, "motor_gucu": motor_gucu,
    }])
    tahmin = model.predict(arac)[0]
    st.success(f"Tahmini fiyat: {tahmin:,.0f} TL")
    st.caption("Bu tahmin, ilan verilerine dayalı bir makine öğrenmesi modeli tarafından üretilmiştir.")
