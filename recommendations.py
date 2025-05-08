import pandas as pd
import numpy as np
from datetime import timedelta

def build_recommendations(df, combined_df, targets, top_n=5, random_n=5, weights=None):
    # Установка весов по умолчанию
    if weights is None:
        default_weight = 0.8 / len(targets) if targets else 0.2  # Равномерное распределение для метрик
        weights = {metric: default_weight for metric in targets}
        weights['competition'] = 0.2  # Фиксированный вес для конкуренции

    # Создаём копию объединённого DataFrame
    forecast_df = combined_df.copy()

    # Вычисляем конкуренцию (используем Hours_streamed как прокси)
    forecast_df['competition'] = forecast_df.get('Hours_streamed_pred', 0) / 1000  # Нормализуем для расчёта

    # Вычисляем Score для каждой игры
    score_components = []
    for metric in targets:
        pred_col = f'{metric}_pred'
        if pred_col in forecast_df.columns:
            score_components.append(weights.get(metric, 0) * forecast_df[pred_col])
    # Вычитаем конкуренцию
    score_components.append(-weights['competition'] * forecast_df['competition'])
    forecast_df['Score'] = sum(score_components)

    # Для пояснений: вычисляем процентный рост Avg_viewers
    games = forecast_df.index
    last_values = {}
    for game in games:
        sub = df[df['Game'] == game].sort_values('ds')
        last_date = sub['ds'].max()
        last_values[game] = {metric: sub[sub['ds'] == last_date][metric].iloc[0] for metric in targets if metric in sub.columns}

    if 'Avg_viewers' in targets:
        forecast_df['Avg_viewers_last'] = [last_values[game].get('Avg_viewers', 0) for game in games]
        forecast_df['Avg_viewers_growth'] = (
            (forecast_df['Avg_viewers_pred'] - forecast_df['Avg_viewers_last']) /
            forecast_df['Avg_viewers_last'] * 100
        ).fillna(0)
    else:
        forecast_df['Avg_viewers_growth'] = 0

    # Ранжируем по Score и выбираем топ-N
    top_games = forecast_df.sort_values('Score', ascending=False).head(top_n)

    # Выбираем случайные игры из оставшихся
    remaining_games = forecast_df.drop(top_games.index)
    if len(remaining_games) >= random_n:
        random_games = remaining_games.sample(n=random_n, random_state=42)
    else:
        random_games = remaining_games

    # Объединяем топовые и случайные игры
    recommended_games = pd.concat([top_games, random_games])

    # Формируем пояснения
    recommendations = []
    last_date = df['ds'].max()
    next_month = (last_date + timedelta(days=31)).strftime('%B %Y')  # Следующий месяц
    for game, row in recommended_games.iterrows():
        growth = row['Avg_viewers_growth'] if 'Avg_viewers' in targets else 0
        competition = row['competition']
        explanation = (
            f"In {next_month}, {game} is expected to see "
            f"{'a growth' if growth >= 0 else 'a decline'} in average audience → "
            f"{'+' if growth >= 0 else ''}{growth:.1f}% Avg_viewers, "
            f"{'low' if competition < 100 else 'high'} competition → "
            f"{'top pick' if game in top_games.index else 'explore this game'}!"
        )
        recommendations.append({
            'Game': game,
            'Score': row['Score'],
            'Explanation': explanation,
            'Category': 'Top Pick' if game in top_games.index else 'Explore'
        })

    return pd.DataFrame(recommendations)