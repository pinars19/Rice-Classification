import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.title("Rice Classification - CNN")
st.write("Pirinç görüntüsü yükleyerek çeşidini tahmin edin.")

model = load_model("rice_cnn.keras")
classes = ["Arborio","Basmati","Ipsala","Jasmine","Karacadag"]

file = st.file_uploader("Görüntü yükle",type=["jpg","jpeg","png"])

if file is not None:
    img = Image.open(file).convert("RGB").resize((128,128))
    st.image(img,caption="Yüklenen görüntü")

    x = np.expand_dims(np.array(img)/255.0,axis=0)
    pred = model.predict(x,verbose=0)[0]

    st.success(f"Tahmin: {classes[np.argmax(pred)]}")
    st.write(f"Güven skoru: {np.max(pred):.2%}")
