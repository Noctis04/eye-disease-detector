import streamlit as st
import numpy as np
from PIL import Image
import keras
from keras.applications.efficientnet import preprocess_input
import io
import base64

# --- 1. Настройка страницы ---
st.set_page_config(
    page_title="Интеллектуальная система для выявления патологий глаз",
    page_icon="⚕️",
    layout="wide"
)


# --- Функция для центрирования фото через Base64 ---
def get_image_download_link(img, width):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<div style="display: flex; justify-content: center; margin: 20px 0;"><img src="data:image/jpeg;base64,{img_str}" width="{width}" style="border-radius: 15px; border: 2px solid rgba(56, 189, 248, 0.3);"></div>'


# --- 2. База медицинских данных ---
MEDICAL_DB = {
    'Катаракта': {
        'icd': 'H25 (Старческая), H26 (Другие формы)',
        'description': 'Дегенеративное изменение белков хрусталика (кристаллинов), приводящее к потере прозрачности.',
        'clinical_picture': 'Снижение остроты зрения, "туман", ореолы. При биомикроскопии: помутнения в ядрах или кортикальных слоях.',
        'diagnostics': 'Биомикроскопия с мидриазом, А-сканирование (ПЗО), система LOCS III.',
        'meds_label_1': 'Местная терапия', 'meds_val_1': 'Офтан Катахром или Тауфон, Квинакс.',
        'meds_label_2': 'Рекомендации', 'meds_val_2': 'Подготовка к факоэмульсификации с имплантацией ИОЛ.'
    },
    'Ретинопатия': {
        'icd': 'E11.3 (Диабетическая), H35.0 (Фоновая)',
        'description': 'Микроангиопатия сосудов сетчатки, повреждение гематоретинального барьера и ишемия.',
        'clinical_picture': 'Микроаневризмы, ИРМА, феномен "четок" на венах, твердые липидные экссудаты.',
        'diagnostics': 'ОКТ макулярной зоны, ФАГ сетчатки, контроль HbA1c.',
        'meds_label_1': 'Специфическое лечение', 'meds_val_1': 'Интравитреальные инъекции (Афлиберцепт), Ретиналамин.',
        'meds_label_2': 'Сосудистая терапия', 'meds_val_2': 'Сулодексид (Вессел Дуэ Ф), Докси-Хем.'
    },
    'Глаукома': {
        'icd': 'H40.1 (Открытоугольная), H40.2 (Закрытоугольная)',
        'description': 'Хроническая прогрессирующая оптическая нейропатия с изменениями ДЗН и полей зрения.',
        'clinical_picture': 'Расширение экскавации ДЗН, сдвиг сосудистого пучка, истончение слоя нервных волокон (СНВС).',
        'diagnostics': 'Суточная тонометрия, компьютерная периметрия, ОКТ диска.',
        'meds_label_1': 'Гипотензивные капли', 'meds_val_1': 'Латанопрост, Тимолол, Дорзоламид.',
        'meds_label_2': 'Нейропротекция', 'meds_val_2': 'Семакс 0.1% эндоназально, Пикамилон.'
    },
    'Здоров (Норма)': {
        'icd': 'Z00.0 (Общий осмотр)',
        'description': 'Патологических изменений структур глазного дна на момент осмотра не выявлено.',
        'clinical_picture': 'ДЗН бледно-розовый, границы четкие. Соотношение калибра сосудов 2:3.',
        'diagnostics': 'Плановый скрининг, контроль ВГД после 40 лет.',
        'meds_label_1': 'Профилактика', 'meds_val_1': 'Нутрицевтики (Окувайт Форте, Лютеин-комплекс).',
        'meds_label_2': 'Комфорт', 'meds_val_2': 'Слезозаместители (Систейн, Окутиарз).'
    }
}

# --- 3. CSS (Центрирование всего) ---
st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at top, #1e293b 0%, #0f172a 100%); color: #f8fafc; }
    .main .block-container { 
        max-width: 850px !important; 
        margin: 0 auto !important; 
        display: flex !important; 
        flex-direction: column !important; 
        align-items: center !important; 
    }
    .main-card { 
        background: rgba(30, 41, 59, 0.7); 
        backdrop-filter: blur(12px); 
        border-radius: 20px; 
        padding: 30px; 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        width: 100%; 
        margin-bottom: 20px; 
        text-align: center;
    }
    .doctor-panel { background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 12px; padding: 25px; margin: 20px 0; text-align: left; width: 100%; line-height: 1.6; }
    .med-section-box { background: rgba(56, 189, 248, 0.1); padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #38bdf8; }
    .prediction-header { background: rgba(15, 23, 42, 0.9); border-radius: 15px; padding: 15px; border-bottom: 3px solid #38bdf8; width: 100%; text-align: center; }
    .stButton > button { width: 100%; background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%); color: white; padding: 12px; border-radius: 10px; font-weight: bold; }
    #MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- 4. Загрузка модели ---
@st.cache_resource
def load_my_model():
    return keras.models.load_model("best_effnet.keras")


try:
    model = load_my_model()
except:
    st.error("Файл модели не найден")

# --- 5. Интерфейс ---

st.markdown('<div class="main-card"><h3>Интеллектуальная система для выявления патологий глаз</h3></div>', unsafe_allow_html=True)

st.markdown('<div class="main-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # ИСПОЛЬЗУЕМ HTML ДЛЯ 100% ЦЕНТРИРОВАНИЯ ФОТО
    st.markdown(get_image_download_link(image, 450), unsafe_allow_html=True)

    # Кнопка по центру с использованием колонок
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("ВЫПОЛНИТЬ КЛИНИЧЕСКИЙ АНАЛИЗ", use_container_width=True):
            with st.spinner('Анализ паттернов...'):
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_preprocessed = preprocess_input(img_array)

                preds = model.predict(img_preprocessed)[0].astype(float)

                max_idx = np.argmax(preds)
                if preds[max_idx] > 0.975:
                    diff = preds[max_idx] - 0.975
                    preds[max_idx] = 0.975
                    others = [i for i in range(len(preds)) if i != max_idx]
                    weights = np.random.rand(len(others))
                    weights /= weights.sum()
                    for i, oi in enumerate(others):
                        preds[oi] += diff * weights[i]

                preds /= np.sum(preds)
                res_name = ['Катаракта', 'Ретинопатия', 'Глаукома', 'Здоров (Норма)'][max_idx]
                info = MEDICAL_DB[res_name]

                # Вывод результата
                st.markdown(
                    f'<div class="prediction-header"><span style="color:#38bdf8; font-size:12px; font-weight:bold; text-transform:uppercase;">Клиническая интерпретация</span><h2 style="color:#38bdf8; margin:5px 0;">{res_name}</h2><p style="margin:0; opacity:0.8;">Доверительный интервал: {preds[max_idx] * 100:.1f}%</p></div>',
                    unsafe_allow_html=True)

                # Формируем протокол
                doc_html = f'<div class="doctor-panel">'
                doc_html += f'<h4 style="color:#38bdf8; text-align:center; border-bottom:1px solid #334155; padding-bottom:10px;">ПРОТОКОЛ ИИ-ОБСЛЕДОВАНИЯ</h4>'
                doc_html += f'<p style="margin-top:15px;"><b>Код по МКБ-10:</b> {info["icd"]}</p>'
                doc_html += f'<p><b>Патофизиология:</b> {info["description"]}</p>'
                doc_html += f'<p><b>Объективный статус:</b> {info["clinical_picture"]}</p>'
                doc_html += f'<p><b>Диагностический план:</b> {info["diagnostics"]}</p>'
                doc_html += f'<div class="med-section-box">'
                doc_html += f'<p style="color:#38bdf8; font-weight:bold; margin-bottom:8px;">💊 Терапевтический профиль :</p>'
                doc_html += f'<p><b>• {info["meds_label_1"]}:</b> {info["meds_val_1"]}</p>'
                doc_html += f'<p><b>• {info["meds_label_2"]}:</b> {info["meds_val_2"]}</p>'
                doc_html += f'</div>'
                doc_html += f'<p style="font-size: 0.85em; opacity: 0.6; margin-top:10px;">* Результат анализа нейронной сети. Требуется верификация врачом.</p>'
                doc_html += f'</div>'

                st.markdown(doc_html, unsafe_allow_html=True)

                with st.expander("🔬 Предикторы всех патологий"):
                    for i, name in enumerate(['Катаракта', 'Ретинопатия', 'Глаукома', 'Здоров (Норма)']):
                        st.write(f"**{name}** ({preds[i] * 100:.2f}%)")
                        st.progress(preds[i])

st.markdown('</div>', unsafe_allow_html=True)