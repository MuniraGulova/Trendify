import streamlit as st
import base64
import os
import warnings
warnings.filterwarnings("ignore")
from pandas.api.types import CategoricalDtype
import pandas as pd
import plotly.express as px
from forecasting import run_naive, run_moving_average, run_arima, run_sarima, run_sarimax, run_lstm, run_prophet, run_orbit
from recommendations import build_recommendations

# Настройка страницы ^-^
st.set_page_config(layout="wide")

# Инициализация session_state для управления отображением рекомендаций
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False

# CSS для стилизации, включая прозрачность таблиц
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    /* Скрываем стандартные элементы Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    body {
        margin: 0;
        padding: 0;
        background-color: black;
        margin-top: 40px;
    }
    .video-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100vh;
        overflow: hidden;
    }
    .bg-video {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        z-index: -1;
    }
    .fon {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        width: 50%;
        max-width: 720px;
        height: auto;
        position: absolute;
        left: 45%;
        top: 27%;
        border-radius: 30px;
        padding: 20px;
        z-index: 1;
    }
    .custom-text {
        color: rgb(255, 255, 255);
        width: 100%;
        font-size: 9px;
        font-family: Orbitron;
        line-height: 40px;
        margin: 0;
    }
    /* Стили для круглой кнопки */
    :root {
        --glitter: url("https://assets.codepen.io/13471/silver-glitter-background.png");
    }
    .sparkles-container {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 2;
    }
    .sparkles {
        --shadows: 0%;
        --shadowl: 0%;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        border-radius: 50%; /* Изменено на 50% для круглой формы */
        background: linear-gradient(
            0deg,
            hsl(210, 20%, 90%) 0%,
            hsl(210, 50%, 80%) 5%,
            hsl(210, 100%, 60%) 15%,
            hsl(210, 50%, 20%) 40%,
            hsl(210, 30%, 30%) 90%
        );
        background-size: 200% 300%;
        background-position: 0% 0%;
        box-shadow: inset 0 0 2px hsl(210, 30%, 20%);
        display: grid;
        grid-template-columns: 1fr;
        grid-template-rows: 1fr;
        place-items: center;
        padding: 0;
        position: relative;
        overflow: hidden;
        transform: translate(0px);
        transition: all 0.5s cubic-bezier(0.77, 0, 0.18, 1);
        box-shadow:
            0 0.25em 0.3em -0.2em hsl(210, 0%, 0%, 0.5),
            0 0.35em 0.75em hsl(210, 0%, 0%, 0.75);
        text-decoration: none;
        width: 60px; /* Фиксированная ширина для круглой формы */
        height: 60px; /* Фиксированная высота для круглой формы */
    }
    .sparkles::before,
    .sparkles::after {
        --gradientPos: 50% 100%;
        content: " ";
        grid-column: 1;
        grid-row: 1;
        width: 100%;
        height: 100%;
        transition: inherit;
    }
    .sparkles:before {
        inset: 0;
        position: absolute;
        transform: translate3d(0, 0, 0.01px);
        border-radius: inherit;
        background-image: var(--glitter), var(--glitter),
            linear-gradient(180deg, hsl(210, 20%, 10%) 0%, hsl(210, 20%, 90%) 80%);
        background-size: 300px 170px, 280px 130px, 200% 200%;
        background-blend-mode: multiply, multiply, overlay;
        background-position: 0px 0px, 0px 0px, var(--gradientPos);
        background-repeat: repeat;
        mix-blend-mode: color-dodge;
        filter: brightness(2) contrast(0.75);
        animation: bubble 20s linear infinite;
        animation-play-state: paused;
        opacity: 0.5;
        box-shadow: inset 0 -8px 10px -7px hsl(210, 70%, 80%, 0.75);
    }
    .sparkles:after {
        background-image: radial-gradient(
                ellipse at center 70%,
                hsl(210, 20%, 99%, 0.8) 5%,
                hsl(210, 100%, 60%, 1) 20%,
                transparent 50%,
                transparent 200%
            ),
            linear-gradient(
                90deg,
                hsl(210, 80%, 10%, 1) -10%,
                transparent 25%,
                transparent 75%,
                hsl(210, 80%, 10%, 1) 110%
            );
        box-shadow: inset 0 0.25em 0.75em rgba(0, 0, 0, 1),
            inset 0 -0.05em 0.2em rgba(255, 255, 255, 0.4),
            inset 0 -1px 3px hsl(210, 80%, 50%, 0.75);
        background-blend-mode: darken;
        background-repeat: no-repeat;
        background-size: 180% 80%, cover;
        background-position: center 220%;
        mix-blend-mode: hard-light;
        filter: blur(5px);
        opacity: 0;
    }
    .sparkles:hover {
        --shadows: 90%;
        --shadowl: 80%;
        background-position: 100% 100%;
        transition: all 0.2s cubic-bezier(0.17, 0.84, 0.44, 1);
        box-shadow:
            0 -0.2em 1.5em hsl(210, 100%, 60%, 0.3),
            0 0.5em 2em hsl(210, 100%, 70%, 0.55),
            0 0.25em 0.3em -0.2em hsl(210, 0%, 0%, 1),
            0 0.35em 0.75em hsl(210, 0%, 0%, 1),
            0 0.25em 0.5em -0.3em hsl(210, 30%, 99%, 1),
            0 0.25em 0.5em hsl(210, 20%, 30%, 0.35),
            inset 0 -2px 5px -2px rgba(255, 255, 255, 0.5);
    }
    .sparkles:hover:before {
        --gradientPos: 50% 50%;
        animation-play-state: running;
        filter: brightness(2) contrast(1);
        box-shadow: inset 0 -5px 10px -4px hsl(210, 70%, 80%, 0.3);
        opacity: 0.8;
    }
    .sparkles:hover:after {
        opacity: 0.8;
        transform: translateY(0px);
    }
    .sparkles span {
        grid-column: 1;
        grid-row: 1;
        background-image: linear-gradient(
            hsl(210, 20%, 85%) 0%,
            hsl(210, 30%, 80%) 19%,
            hsl(210, 40%, 75%) 30%,
            hsl(210, 20%, 98%) 43%,
            hsl(210, 100%, 60%, 1) 51%,
            hsl(210, 50%, 85%, 1) 52%,
            rgb(255, 255, 255) 100%
        );
        background-size: 1em 3.45em;
        color: rgb(214, 222, 226);
        -webkit-text-fill-color: transparent;
        -webkit-background-clip: text;
        filter: drop-shadow(0 0 0.05em rgba(128, 128, 128, 0.5)) drop-shadow(0 0.05em 0.05em rgba(128, 128, 128, 0.5));
        transition-timing-function: inherit;
        transition-duration: inherit;
        transition-delay: 0s;
        padding: 0; /* Убраны отступы для центрирования текста */
        transform: translateY(0);
        z-index: 10;
        font-size: 0.8rem; /* Уменьшен размер текста для круглой кнопки */
        text-align: center;
    }
    .sparkles:hover span {
        background-position-y: -100%;
    }
    .sparkles:active {
        transform: translateY(0.075em);
        box-shadow:
            0 -0.2em 1.5em hsl(210, 100%, 60%, 0.4),
            0 0.5em 2em hsl(210, 100%, 70%, 0.65),
            0 0.15em 0.3em -0.2em hsl(210, 0%, 0%, 1),
            0 0.25em 0.75em hsl(210, 0%, 0%, 1),
            0 0.25em 0.5em -0.3em hsl(210, 30%, 99%, 1),
            0 0.25em 0.5em hsl(210, 20%, 30%, 0.45),
            inset 0 -2px 5px -2px rgba(255, 255, 255, 0.65);
        transition-duration: 0.1s;
    }
    .sparkles:active:before {
        opacity: 1;
        filter: brightness(3) contrast(0.75);
        animation-duration: 8s;
    }
    .sparkles:active:after {
        filter: brightness(1.35) contrast(0.8) blur(5px);
    }
    @keyframes bubble {
        0% {
            background-position: 0px 340px, 0px 130px, var(--gradientPos);
        }
        100% {
            background-position: 0px 0px, 0px 0px, var(--gradientPos);
        }
    }
    @keyframes neon-glow {
        0% { text-shadow: 0 0 5px hsl(210, 100%, 60%), 0 0 10px hsl(210, 100%, 60%); }
        50% { text-shadow: 0 0 10px hsl(210, 100%, 60%), 0 0 20px hsl(210, 100%, 60%); }
        100% { text-shadow: 0 0 5px hsl(210, 100%, 60%), 0 0 10px hsl(210, 100%, 60%); }
    }
    /* Стили для боковой панели */
    .stSidebar {
        background: rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(15px);
        color: rgb(214, 222, 226);
        padding: 20px;
        width: 300px;
        box-shadow: 0 0 20px hsl(228, 87%, 50%);
    }
    /* Стили для заголовка "Настройки" */
    .stSidebar h1 {
        color: hsl(210, 100%, 60%);
        text-shadow: 0 0 5px hsl(210, 100%, 60%), 0 0 10px hsl(210, 100%, 60%), 0 0 15px hsl(230, 93%, 58%);
        text-transform: uppercase;
        font-family: Orbitron;
        font-size: 35px;
        margin-bottom: 15px;
        animation: neon-glow 1.5s ease-in-out infinite;
    }
    /* Стили для подзаголовков */
    .stSidebar h2 {
        color: hsl(210, 100%, 60%);
        text-shadow: 0 0 5px hsl(210, 100%, 60%), 0 0 10px hsl(210, 100%, 60%);
        font-family: Orbitron;
    }
    /* Стили для меток виджетов */
    [data-testid="stSidebar"] label,
    .stSelectbox label,
    .stNumberInput label,
    .stCheckbox label,
    .stTextInput label {
        font-family: Orbitron;
        font-size: 9px !important;
        color: rgb(214, 222, 226);
        text-shadow: 0 0 3px hsl(212, 99%, 40%);
    }
    /* Стили для виджетов Streamlit */
    .stSelectbox select {
        background-color: rgba(255, 255, 255, 0.1);
        color: rgb(214, 222, 226);
        border: 2px solid hsl(210, 100%, 60%);
        border-radius: 5px;
        font-family: Orbitron;
        width: 100%;
        padding: 5px;
        box-shadow: 0 0 5px hsl(210, 100%, 60%), 0 0 10px hsl(210, 100%, 60%);
    }
    .stSelectbox select:hover {
        box-shadow: 0 0 10px hsl(210, 100%, 60%), 0 0 20px hsl(210, 100%, 60%);
    }
    .stNumberInput input {
        background-color: rgba(255, 255, 255, 0.1);
        color: rgba(214, 222, 226, 0.93);
        border: 2px solid hsl(210, 100%, 60%);
        border-radius: 5px;
        font-family: Orbitron;
        width: 100%;
        padding: 8px;
        box-shadow: 0 0 5px hsl(210, 100%, 60%), 0 0 10px hsl(210, 96%, 52%);
    }
    .stNumberInput input:hover {
        box-shadow: 0 0 10px hsl(210, 100%, 60%), 0 0 20px hsl(210, 100%, 60%);
    }
    .stCheckbox label {
        color: rgb(214, 222, 226);
        font-family: Orbitron;
        font-size: 9px !important;
        text-shadow: 0 0 3px hsl(210, 100%, 60%);
    }
    .stCheckbox input {
        accent-color: hsl(210, 100%, 60%);
    }
    .stCheckbox input:checked {
        box-shadow: 0 0 5px hsl(210, 100%, 60%), 0 0 10px hsl(210, 94%, 55%);
    }
    .stCheckbox input:hover {
        box-shadow: 0 0 5px hsl(210, 100%, 60%);
    }
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.1);
        color: rgb(214, 222, 226);
        border: 2px solid hsl(210, 100%, 60%);
        border-radius: 5px;
        font-family: Orbitron;
        width: 100%;
        padding: 8px;
        box-shadow: 0 0 5px hsl(210, 81%, 48%), 0 0 10px hsl(231, 91%, 36%);
    }
    .stTextInput input:hover {
        box-shadow: 0 0 10px hsl(210, 89%, 49%), 0 0 20px hsl(231, 82%, 48%);
    }
    /* Тень текста для меток и подзаголовков */
    .stSidebar label,
    .stSidebar h2,
    .stSidebar h3,
    .stSidebar p {
        text-shadow: 0 0 3px hsl(212, 99%, 40%);
    }
    /* Стили для разделителей */
    .stSidebar hr {
        height: 1px;
        background: linear-gradient(90deg,
            transparent,
            rgba(192, 192, 192, 0.6),
            rgba(255, 255, 255, 0.8),
            rgba(192, 192, 192, 0.6),
            transparent);
        margin: 20px 0;
        border: none;
        box-shadow: 0 1px 2px rgba(255, 255, 255, 0.1);
    }
    /* Стили для таблиц (увеличиваем прозрачность) */
    [data-testid="stDataFrame"] {
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(5px);
        color: rgb(255, 255, 255);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 5px;
    }
    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] td {
        color: rgb(255, 255, 255);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Функция для кодирования видео в base64
def get_base64_video(file_path):
    if not os.path.exists(file_path):
        st.error(f"Видео не найдено по пути: {file_path}")
        return None
    with open(file_path, "rb") as video_file:
        encoded = base64.b64encode(video_file.read()).decode()
    return f"data:video/mp4;base64,{encoded}"

video_path = "static/img/live.mp4"
video_base64 = get_base64_video(video_path)

# Видео и контент
if video_base64:
    st.markdown(f"""
    <div class="video-container">
        <video class="bg-video" autoplay=autoplay loop muted playsinline>
            <source src="{video_base64}" type="video/mp4">
            Ваш браузер не поддерживает видео.
        </video>
        <div class="sparkles-container">
            <button class="sparkles" onclick="document.getElementById('recommend-button').click()">
            </button>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Скрытая кнопка Streamlit для обработки клика
if st.button("Show Recommendations", key="recommend-button"):
    st.session_state.show_recommendations = True

# Загрузка данных
df = pd.read_csv('clean_twitch_data.csv', encoding_errors='ignore')
df = df.dropna(subset=['Game'])
df['ds'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
games = df['Game'].unique()

# Панель настроек с использованием Streamlit sidebar
with st.sidebar:
    st.header("Settings")
    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox("Select month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], index=4)
    with col2:
        year = st.selectbox("Select year", ["2020", "2021", "2022", "2023", "2024", "2025"], index=5)
    st.divider()
    model_choice = st.selectbox("Select forecasting model", ["Naive", "Moving Average", "ARIMA", "SARIMA", "SARIMAX", "LSTM", "Prophet", "Orbit"])
    st.divider()
    top_recommendations = st.number_input("Number of top recommendations", min_value=1, max_value=50, value=10)
    random_recommendations = st.number_input("Number of random recommendations", min_value=0, max_value=50, value=5)
    st.divider()
    st.subheader("Additional metrics")
    avg_viewers = st.checkbox("Avg_viewers", value=True)
    peak_viewers = st.checkbox("Peak_viewers")
    avg_viewer_ratio = st.checkbox("Avg_viewer_ratio")
    hours_watched = st.checkbox("Hours_watched")
    hours_streamed = st.checkbox("Hours_streamed")
    st.divider()
    st.subheader("Recommendation Weights")
    # Поля ввода весов для всех метрик
    weight_avg_viewers = st.number_input("Weight for Avg_viewers", min_value=0.0, max_value=1.0, value=0.4, step=0.01) if avg_viewers else 0.0
    weight_peak_viewers = st.number_input("Weight for Peak_viewers", min_value=0.0, max_value=1.0, value=0.3, step=0.01) if peak_viewers else 0.0
    weight_avg_viewer_ratio = st.number_input("Weight for Avg_viewer_ratio", min_value=0.0, max_value=1.0, value=0.2, step=0.01) if avg_viewer_ratio else 0.0
    weight_hours_watched = st.number_input("Weight for Hours_watched", min_value=0.0, max_value=1.0, value=0.05, step=0.01) if hours_watched else 0.0
    weight_hours_streamed = st.number_input("Weight for Hours_streamed", min_value=0.0, max_value=1.0, value=0.05, step=0.01) if hours_streamed else 0.0
    weight_competition = st.number_input("Weight for Competition", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    # Проверка суммы весов
    total_weight = (
        weight_avg_viewers + weight_peak_viewers + weight_avg_viewer_ratio +
        weight_hours_watched + weight_hours_streamed + weight_competition
    )
    if abs(total_weight - 1.0) > 0.01:
        st.warning("Sum of weights should equal 1. Current sum: {:.2f}".format(total_weight))
    st.divider()
    st.subheader("Game for analysis")
    selected_game = st.selectbox("Выберите игру", games)

# Определяем столбцы для агрегирования
first_cols = ['Month', 'Year', 'Rank']
mean_cols = ['Avg_viewers', 'Peak_viewers', 'Avg_viewer_ratio', 'Hours_watched', 'Hours_streamed']
agg_dict = {col: 'first' if col in first_cols else 'mean' for col in df.columns if col in first_cols + mean_cols}
df = df.groupby(['ds', 'Game']).agg(agg_dict).reset_index()

# Определяем цели и игры :)
targets = [metric for metric, selected in [
    ('Avg_viewers', avg_viewers),
    ('Peak_viewers', peak_viewers),
    ('Avg_viewer_ratio', avg_viewer_ratio),
    ('Hours_watched', hours_watched),
    ('Hours_streamed', hours_streamed)
] if selected]

# Глобальные переменные
global_last_date = df['ds'].max()
next_month = global_last_date + pd.offsets.MonthBegin(1)

# Формирование словаря весов
weights = {
    'Avg_viewers': weight_avg_viewers,
    'Peak_viewers': weight_peak_viewers,
    'Avg_viewer_ratio': weight_avg_viewer_ratio,
    'Hours_watched': weight_hours_watched,
    'Hours_streamed': weight_hours_streamed,
    'competition': weight_competition
}

# Выбор модели
model_functions = {
    "Naive": run_naive,
    "Moving Average": run_moving_average,
    "ARIMA": run_arima,
    "SARIMA": run_sarima,
    "SARIMAX": run_sarimax,
    "LSTM": run_lstm,
    "Prophet": run_prophet,
    "Orbit": run_orbit
}
selected_model = model_functions.get(model_choice, run_naive)
df_model = selected_model(df, targets)

# Анализ игры
with st.container():
    with st.expander('❄️ Data', expanded=False):
        st.write('Games')
        st.dataframe(df)
    with st.expander('🎮 Selected Game', expanded=False):
        game_data = df[df['Game'] == selected_game].sort_values('ds')
        st.dataframe(game_data)

    fig = px.line(game_data, x='ds', y='Avg_viewers', title=f"{selected_game} - Среднее количество зрителей")
    st.plotly_chart(fig)
    if df_model is not None and selected_game in df_model.index:
        predictions = df_model.loc[selected_game]
        st.subheader("Прогнозы на следующий месяц")
        for metric in targets:
            pred_col = f'{metric}_pred'
            if pred_col in predictions:
                st.write(f"{metric}: {predictions[pred_col]}")

# Рекомендации
if st.session_state.show_recommendations:
    recommendations = build_recommendations(df, df_model, targets, top_n=top_recommendations, random_n=random_recommendations, weights=weights)
    with st.container():
        st.header("Рекомендуемые игры")
        st.dataframe(recommendations)

    with st.expander('🕐 Forecast Data', expanded=False):
        st.dataframe(df_model)
