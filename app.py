import os
import pycountry
import base64
import time
import streamlit as st
import pandas as pd
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime
from PIL import Image
import plotly.express as px
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components


# ---------------------------
# Configuración de la página
# ---------------------------
st.set_page_config(page_title='Vuela Lejos — Recomendador',
                   layout='wide',
                   initial_sidebar_state='expanded')

# ---------------------------
# CSS global (fuentes + paleta profesional + cards)
# ---------------------------
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@400;600&display=swap" rel="stylesheet">

    <style>
    /* Fondo y fuente */
    .stApp {
        background: linear-gradient(180deg, #f5fbff 0%, #eef7fb 40%, #ffffff 100%);
        color: #0b2840;
        font-family: 'Inter', 'Poppins', sans-serif;
    }

    /* Títulos principales */
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #08314b;
        letter-spacing: 0.2px;
    }
    .subtitle {
        color: #0b4b66;
        margin-top: -8px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b4b66 0%, #135f83 100%);
        color: #ffffff;
    }
    section[data-testid="stSidebar"] .stTextInput > label,
    section[data-testid="stSidebar"] .stSelectbox > label {
        color: #ffffff;
    }
    section[data-testid="stSidebar"] .css-1d391kg { color: #ffffff; }

        /* Etiquetas del slider dentro del sidebar */
    section[data-testid="stSidebar"] .stSlider > label {
        color: #ffffff !important;
    }


    /* Botones primarios */
    .stButton > button {
        background: linear-gradient(90deg,#1271c7,#1fa2ff);
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 700;
        box-shadow: 0 6px 18px rgba(17,82,122,0.12);
        border: none;
    }
    .stButton > button:hover { transform: translateY(-2px); transition: 0.15s; }

    /* Inputs pulidos */
    input, textarea, select {
        border-radius: 8px !important;
        border: 1px solid rgba(10,40,60,0.08) !important;
        padding: 8px !important;
    }

    /* Card visual para recomendaciones */
    .card {
        border-radius: 12px;
        padding: 14px;
        background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(247,251,255,0.85));
        box-shadow: 0 6px 20px rgba(14,50,75,0.06);
        border: 1px solid rgba(13,67,94,0.05);
        margin-bottom: 14px;
    }
    .card-title { font-weight: 700; color: #08314b; font-size: 18px; margin-bottom: 6px; }
    .chip { display:inline-block; padding:6px 10px; border-radius:999px; background:#e6f3ff; color:#0b4b66; font-weight:600; margin-right:6px; }

    /* Footer */
    .footer { text-align:center; color: #0b4b66; margin-top: 24px; padding-top: 8px; opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Claves
# ---------------------------
API_KEY = os.environ.get('OPENWEATHER_KEY', 'f0519d60fceacc3438a9cb4d03bb9704')
UNSPLASH_ACCESS_KEY = os.environ.get('UNSPLASH_ACCESS_KEY', '')

# ---------------------------
# Utilidades
# ---------------------------
def country_name_to_iso2(name):
    try:
        country = pycountry.countries.lookup(name)
        return country.alpha_2
    except Exception:
        return None

def obtener_clima(ciudad, pais_nombre):
    pais = country_name_to_iso2(pais_nombre)
    if not pais or not ciudad or pd.isna(ciudad):
        return "No disponible"
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={ciudad},{pais}&appid={API_KEY}&units=metric&lang=es"
        respuesta = requests.get(url, timeout=6)
        datos = respuesta.json()
        if respuesta.status_code != 200:
            return "No disponible"
        descripcion = datos['weather'][0]['description'].capitalize()
        temperatura = datos['main']['temp']
        return f"{descripcion}, {temperatura} °C"
    except Exception:
        return "No disponible"

def generar_pdf(nombre_usuario, recomendaciones_df):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Portada estilizada
    pdf.setFillColorRGB(0.03,0.2,0.33)
    pdf.setFont('Helvetica-Bold', 30)
    pdf.drawCentredString(width / 2, height - 120, 'Vuela Lejos')
    pdf.setFont('Helvetica', 12)
    pdf.drawCentredString(width / 2, height - 150, f'Recomendaciones personalizadas para {nombre_usuario}')
    fecha = datetime.now().strftime('%d/%m/%Y')
    pdf.setFont('Helvetica', 10)
    pdf.drawCentredString(width / 2, height - 170, f'Generado: {fecha}')
    pdf.showPage()

    pdf.setFont('Helvetica-Bold', 18)
    pdf.drawString(48, height - 60, 'Destinos sugeridos')
    y = height - 90
    for idx, row in recomendaciones_df.iterrows():
        if y < 120:
            pdf.showPage()
            y = height - 60
        pdf.setFont('Helvetica-Bold', 12)
        pdf.drawString(48, y, f"{row.get('destination', '')} — {row.get('country', '')}")
        y -= 16
        pdf.setFont('Helvetica', 10)
        pdf.drawString(60, y, f"Categoría: {row.get('category','')}")
        y -= 12
        pdf.drawString(60, y, f"Actividades: {str(row.get('activities',''))[:120]}")
        y -= 12
        pdf.drawString(60, y, f"Descripción: {str(row.get('description',''))[:160]}")
        y -= 20
        pdf.setLineWidth(0.3)
        pdf.line(48, y, width - 48, y)
        y -= 18

    pdf.save()
    buffer.seek(0)
    return buffer

# ---------------------------
# Carga de datos
# ---------------------------
data_path = os.path.join('processed', 'cleaned_data.csv')
if not os.path.exists(data_path):
    st.sidebar.error('No se encontró el dataset procesado. Ejecuta el script de generación antes.')
    st.stop()

try:
    data = pd.read_csv(data_path)
except Exception as e:
    st.sidebar.error(f'Error al leer el dataset: {e}')
    st.stop()

# Normalizaciones mínimas
for col in ['activities', 'description', 'category', 'destination', 'country']:
    if col in data.columns:
        data[col] = data[col].fillna('')
    else:
        data[col] = ''

data['text_features'] = data['activities'].astype(str) + ' ' + data['description'].astype(str) + ' ' + data['category'].astype(str)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['text_features'])

# ---------------------------
# Sidebar (perfil)
# ---------------------------
with st.sidebar:
    st.markdown("<h2 style='color:white;margin-bottom:2px'>Perfil del viajero</h2>", unsafe_allow_html=True)
    nombre = st.text_input('Nombre completo')
    edad = st.text_input('Edad (años)')
    st.markdown('---')
    st.markdown("<h4 style='color:white;margin-bottom:6px'>Preferencias</h4>", unsafe_allow_html=True)
    tipo_viajero = st.selectbox('Tipo de viajero', ['Aventurero', 'Cultural', 'Familiar', 'Relajado', 'Romántico'])
    presupuesto = st.selectbox('Presupuesto', ['Económico', 'Moderado', 'Alto'])
    tiempo = st.slider('Duración (días)', 1, 30, 7)
    pais = st.selectbox('Destino preferido', ['Cualquiera'] + sorted(data['country'].dropna().unique()))
    st.markdown('---')
    buscar = st.button('Buscar recomendaciones')

# Validación edad
if edad:
    try:
        edad_int = int(edad)
        if edad_int < 18:
            st.sidebar.warning('Debes ser mayor de edad para usar esta aplicación.')
    except ValueError:
        st.sidebar.error('Edad inválida (usa números).')

# ---------------------------
# Portada (title + slideshow)
# ---------------------------
# Imagenes por defecto para slideshow
SLIDES = [
    "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=1200&q=80&auto=format&fit=crop",
    "https://plus.unsplash.com/premium_photo-1664300792059-863ccfe55932?w=1200&q=80&auto=format&fit=crop",
    "https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?w=1200&q=80&auto=format&fit=crop",
    "https://images.unsplash.com/photo-1603565816030-6b389eeb23cb?w=1200&q=80&auto=format&fit=crop"
]

# ---------------------------
# Header + Slideshow (solo se muestran si NO hay 'resultado' en session_state)
# ---------------------------

# ---------------------------
# Lógica de búsqueda
# ---------------------------
if buscar:
    descripcion_tipo = {
        'Aventurero': 'montaña naturaleza aventura senderismo',
        'Cultural': 'museos historia arte tradiciones',
        'Familiar': 'actividades para niños parques experienciales',
        'Relajado': 'playa descanso bienestar spa',
        'Romántico': 'escapada pareja cenas atardeceres'
    }
    descripcion_presupuesto = {
        'Económico': 'bajo costo económico',
        'Moderado': 'precio medio equilibrado',
        'Alto': 'lujo confort servicios premium'
    }

    preferencia_texto = f"{descripcion_tipo[tipo_viajero]} {descripcion_presupuesto[presupuesto]} viaje {tiempo} días"
    user_vector = vectorizer.transform([preferencia_texto])
    similitudes = cosine_similarity(user_vector, X).flatten()
    data['score'] = similitudes
    resultado = data.copy()

    if pais and pais != 'Cualquiera':
        resultado = resultado[resultado['country'] == pais]

    resultado = resultado.sort_values(by='score', ascending=False)
    resultado = resultado.drop_duplicates(subset=['destination'], keep='first')
    resultado = resultado.head(10)  # mostrar algo más para UI; luego limitamos

    # Guardar en session_state
    st.session_state['resultado'] = resultado
    st.session_state['nombre'] = nombre or 'Usuario'
    st.session_state['search_meta'] = {'tipo': tipo_viajero, 'presupuesto': presupuesto, 'tiempo': tiempo}

# Mostrar portada (título + párrafo + slideshow) únicamente cuando NO hay resultados
if 'resultado' not in st.session_state:
    # Contenedor superior (título y párrafo, ancho completo)
    st.markdown(
        """
        <div style="width:100%; padding: 8px 8px 0 8px;">
          <div style="max-width:1200px; margin: 0 auto;">
            <div class='main-title' style='font-size:32px; margin-bottom:6px;'>Vuela Lejos — Recomendador de destinos</div>
            <div class='subtitle' style='font-size:15px; margin-bottom:8px;'>Encuentra destinos que se adapten a tu estilo, presupuesto y tiempo.</div>
            <hr style="opacity:0.06; margin-top:6px; margin-bottom:16px;">
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Slideshow: debajo del texto, centrado y ocupando el ancho máximo (respetando padding)
    slideshow_html = f"""
    <style>
      #slideshow-wrap {{
        width:100%;
        display:flex;
        justify-content:center;
        padding: 0 8px 16px 8px;
      }}
      #slideshow-container {{
        width:100%;
        max-width:1200px;
        position: relative;
        height: 520px; /* ajusta aquí si quieres más/menos alto */
        overflow: hidden;
        border-radius: 12px;
        box-shadow: 0 8px 30px rgba(10,30,50,0.06);
      }}
      .slide-image {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 520px; /* mantener igual que container */
        object-fit: cover;
        border-radius: 12px;
        transition: transform 0.7s ease-in-out;
        will-change: transform, opacity;
      }}
      .pos-center {{ transform: translateX(0%); z-index: 2; }}
      .pos-right  {{ transform: translateX(100%); z-index: 1; }}
      .pos-left   {{ transform: translateX(-100%); z-index: 0; }}
      #slide-caption {{
        position: absolute;
        left: 16px;
        bottom: 16px;
        padding: 8px 12px;
        background: rgba(0,0,0,0.35);
        color: white;
        border-radius: 8px;
        font-family: Inter, sans-serif;
        z-index: 3;
      }}
    </style>

    <div id="slideshow-wrap">
      <div id="slideshow-container" aria-live="polite">
        <img id="imgA" class="slide-image pos-center" src="{SLIDES[0] if len(SLIDES)>0 else ''}" alt="slide A">
        <img id="imgB" class="slide-image pos-right" src="{SLIDES[1] if len(SLIDES)>1 else SLIDES[0] if len(SLIDES)>0 else ''}" alt="slide B">
        <div id="slide-caption">Relájate — Explora playas y escapes</div>
      </div>
    </div>

    <script>
    (function() {{
      const slides = {SLIDES};
      const captions = [
        "Relájate — Explora playas y escapes",
        "Aventura — Naturaleza y rutas inolvidables",
        "Diversión — Escapadas cortas y urbanas",
        "Cultura — Ciudades llenas de historia"
      ];

      if (!slides || slides.length === 0) return;

      const imgA = document.getElementById('imgA');
      const imgB = document.getElementById('imgB');
      const captionEl = document.getElementById('slide-caption');
      const container = document.getElementById('slideshow-container');

      let current = 0;
      let currentEl = imgA;
      let incomingEl = imgB;
      let isAnimating = false;
      let timerId = null;
      const intervalMs = 4000;

      function prepareIncoming(idx) {{
        incomingEl.style.transition = 'none';
        incomingEl.classList.remove('pos-left','pos-center');
        incomingEl.classList.add('pos-right');
        incomingEl.src = slides[idx % slides.length];
        void incomingEl.offsetWidth;
        incomingEl.style.transition = '';
      }}

      function startTimer() {{
        if (timerId) return;
        timerId = setInterval(() => nextSlide(), intervalMs);
      }}

      function stopTimer() {{
        if (!timerId) return;
        clearInterval(timerId);
        timerId = null;
      }}

      function nextSlide() {{
        if (isAnimating) return;
        isAnimating = true;

        const following = (current + 1) % slides.length;
        prepareIncoming(following);

        currentEl.classList.remove('pos-center');
        currentEl.classList.add('pos-left');

        incomingEl.classList.remove('pos-right');
        incomingEl.classList.add('pos-center');

        captionEl.textContent = captions[following % captions.length] || '';

        const onTransitionEnd = function(e) {{
          if (e.propertyName && e.propertyName !== 'transform') return;

          const old = currentEl;
          currentEl = incomingEl;
          incomingEl = old;

          current = following;
          prepareIncoming((current + 1) % slides.length);

          isAnimating = false;
        }};

        incomingEl.addEventListener('transitionend', onTransitionEnd, {{ once: true }});
      }}

      if (slides.length === 1) {{
        captionEl.textContent = captions[0] || '';
        return;
      }}

      current = 0;
      currentEl = imgA;
      incomingEl = imgB;
      currentEl.src = slides[0];
      prepareIncoming(1);
      captionEl.textContent = captions[0] || '';

      startTimer();

      container.addEventListener('mouseenter', () => stopTimer());
      container.addEventListener('mouseleave', () => startTimer());
      container.addEventListener('focusin', () => stopTimer());
      container.addEventListener('focusout', () => startTimer());

      window._vl_slideshow = {{ nextSlide, startTimer, stopTimer }};
    }})();
    </script>
    """
    components.html(slideshow_html, height=640)

# ---------------------------
# Presentación de resultados
# ---------------------------
st.markdown("")  # pequeño separador
if 'resultado' in st.session_state:
    resultado = st.session_state['resultado']
    nombre_usuario = st.session_state.get('nombre', 'Usuario')
    meta = st.session_state.get('search_meta', {})

    st.markdown(f"### Hola {nombre_usuario} — Recomendaciones encontradas")
    st.markdown(f"**Tipo:** {meta.get('tipo','-')} · **Presupuesto:** {meta.get('presupuesto','-')} · **Tiempo:** {meta.get('tiempo','-')} días")
    st.markdown("---")

    # Mostrar top 5 como cards visuales
    display_results = resultado.head(5).copy().reset_index(drop=True)
    for idx, row in display_results.iterrows():
        cols = st.columns([1, 2])
        with cols[0]:
            img_url = row.get('image_url') if pd.notna(row.get('image_url')) and row.get('image_url') else SLIDES[idx % len(SLIDES)]
            st.image(img_url, caption=f"{row.get('destination','')} — {row.get('country','')}", use_container_width=True)
        with cols[1]:
            # Obtener clima ANTES de construir la card
            clima_actual = obtener_clima(row.get('destination',''), row.get('country',''))

            # HTML card WITH CLIMA INCLUDED
            card_html = f"""
            <div class="card">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                <div class="card-title">{row.get('destination','')} — {row.get('country','')}</div>
                <div style="margin-bottom:6px;"><span class="chip">{row.get('category','')}</span></div>
                </div>
            </div>

            <div style="color:#11435a;margin-top:8px">
                {str(row.get('description',''))[:280]}
            </div>

            <div style="margin-top:10px; font-size:13px; color:#234b60;">
                <strong>Actividades:</strong> {str(row.get('activities',''))[:220]}
            </div>

            <div style="margin-top:10px;display:flex;gap:8px;align-items:center;font-size:12px;color:#5a7480;">
                <strong>Costo aprox:</strong> ${row.get('average_cost','N/A')}
            </div>

            <div style="margin-top:12px;padding:8px;background:#e8f5ff;border-radius:8px;
                        font-size:13px;color:#0d3a4a;">
                <strong>Clima actual:</strong> {clima_actual}
            </div>

            </div>
            """

            st.markdown(card_html, unsafe_allow_html=True)


    # Descarga PDF
    pdf_buffer = generar_pdf(nombre_usuario, resultado.head(10))
    st.download_button(label='📄 Descargar recomendaciones (PDF)', data=pdf_buffer,
                       file_name=f'recomendaciones_{nombre_usuario.replace(" ","_")}.pdf', mime='application/pdf')

    # Dashboard simplificado a continuación (metrics, tabla y gráfico)
    st.markdown("---")
    st.markdown("## Panel rápido")
    left, right = st.columns([1, 2])
    with left:
        top_dest = resultado.iloc[0] if len(resultado) > 0 else None
        if top_dest is not None:
            st.metric(label="Top recomendado", value=f"{top_dest['destination']} ({top_dest['country']})")
        st.metric(label="Resultados encontrados", value=len(resultado))
    with right:
        st.subheader('Top destinos (tabla)')
        st.dataframe(resultado[['destination', 'country', 'category', 'score']].head(10), use_container_width=True)

    st.subheader('Distribución por categoría')
    fig_cat = px.histogram(resultado, x='category', title='Categorías en resultados', labels={'category': 'Categoría'})
    st.plotly_chart(fig_cat, use_container_width=True)

    # Mostrar clima por destino (tabla)
    st.subheader('Clima de los destinos')
    climas = []
    for _, row in resultado.iterrows():
        climas.append({'Destino': row['destination'], 'País': row['country'], 'Clima': obtener_clima(row['destination'], row['country'])})
    df_clima = pd.DataFrame(climas)
    st.dataframe(df_clima, use_container_width=True)

    # Mapa si hay lat/lon (intentar detectar columnas comunes)
    lat_col = None
    lon_col = None
    for c in ['latitude','lat','Latitude','LAT','latitud','latitud_decimal']:
        if c in data.columns:
            lat_col = c
            break
    for c in ['longitude','lon','Longitude','LON','longitud','longitud_decimal']:
        if c in data.columns:
            lon_col = c
            break

    
# ---------------------------
# Footer
# ---------------------------
st.markdown("""
<div class="footer">
    <hr style="opacity:0.06"/>
    Vuela Lejos · 2025
</div>
""", unsafe_allow_html=True)
