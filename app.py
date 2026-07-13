import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Kullanacagimiz sutunlar (artik seri ve model de dahil)
KATEGORIK = ["konum", "marka", "seri", "model", "vites_tipi", "yakit_tipi", "kasa_tipi", "cekis"]
SAYISAL   = ["yil", "kilometre", "motor_hacmi", "motor_gucu"]

# Veri okunur ve model uygulama acilirken bir kez egitilir (cache).
@st.cache_resource
def veri_ve_model():
    df = pd.read_csv("cars1.csv")
    df = df[KATEGORIK + SAYISAL + ["fiyat"]].dropna()
    df = df[(df["fiyat"] > 50000) & (df["fiyat"] < 15000000)]  # hatali ilanlari at
    X = df[KATEGORIK + SAYISAL]
    y = df["fiyat"]
    onisleme = ColumnTransformer(
        # min_frequency: nadir seri/model degerlerini gruplar, bellegi korur
        [("kat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), KATEGORIK)],
        remainder="passthrough"
    )
    pipe = Pipeline([
        ("onisleme", onisleme),
        ("model", RandomForestRegressor(n_estimators=40, max_depth=18,
                                        random_state=42, n_jobs=-1))
    ])
    pipe.fit(X, y)
    return df, pipe

df, pipe = veri_ve_model()

st.title("🚗 Araba Fiyat Tahmini")
st.write("Aracın özelliklerini seç, tahmini fiyatı gör. (Türkiye 2. el araç verisi ile eğitilmiştir.)")

def sirala(seri):
    return sorted(seri.dropna().astype(str).unique().tolist())

col1, col2 = st.columns(2)
with col1:
    konum = st.selectbox("Şehir", sirala(df["konum"]))
    # Kademeli menuler: marka -> seri -> model
    marka = st.selectbox("Marka", sirala(df["marka"]))
    seri_secenek = sirala(df[df["marka"] == marka]["seri"])
    seri = st.selectbox("Seri", seri_secenek)
    model_secenek = sirala(df[(df["marka"] == marka) & (df["seri"] == seri)]["model"])
    secili_model = st.selectbox("Model", model_secenek)
    yakit_tipi = st.selectbox("Yakıt tipi", sirala(df["yakit_tipi"]))
with col2:
    vites_tipi = st.selectbox("Vites tipi", sirala(df["vites_tipi"]))
    kasa_tipi  = st.selectbox("Kasa tipi", sirala(df["kasa_tipi"]))
    cekis      = st.selectbox("Çekiş", sirala(df["cekis"]))
    yil        = st.number_input("Model yılı", min_value=1990, max_value=2026, value=2018)
    kilometre  = st.number_input("Kilometre", min_value=0, value=100000, step=5000)

col3, col4 = st.columns(2)
with col3:
    motor_hacmi = st.number_input("Motor hacmi (cc)", min_value=800, value=1600, step=100)
with col4:
    motor_gucu  = st.number_input("Motor gücü (bg)", min_value=40, value=120, step=10)

if st.button("Fiyatı Tahmin Et", type="primary"):
    arac = pd.DataFrame([{
        "konum": konum, "marka": marka, "seri": seri, "model": secili_model,
        "vites_tipi": vites_tipi, "yakit_tipi": yakit_tipi,
        "kasa_tipi": kasa_tipi, "cekis": cekis,
        "yil": yil, "kilometre": kilometre,
        "motor_hacmi": motor_hacmi, "motor_gucu": motor_gucu,
    }])
    tahmin = pipe.predict(arac)[0]
    st.success(f"Tahmini fiyat: {tahmin:,.0f} TL")
    st.caption(f"{marka} {seri} • {yil} • {kilometre:,.0f} km")
