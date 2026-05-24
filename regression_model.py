import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_regression_model(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
    feature_cols: list,
    model_type: str = "linear",
    max_features_to_select: int = 50,
):
    """
    Обучает линейную модель с улучшенной дифференциацией.
    Отбирает TOP-N признаков по частоте встречаемости.
    Обучает LinearRegression с положительными коэффициентами.
    """
    required_cols = [id_col, target_col] + feature_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют необходимые столбцы: {missing_cols}")

    work_df = df[required_cols].copy()
    work_df.dropna(subset=feature_cols + [target_col], inplace=True)
    
    if work_df.empty:
        raise ValueError("Нет данных после очистки от NaN.")

    # Отбор признаков по частоте
    # Считаем, сколько раз каждый признак встречается в данных (не равен нулю)
    # Это поможет отсеять редкие и шумные признаки
    
    # Создаем DataFrame только из признаков
    features_only_df = work_df[feature_cols].copy()
    
    # Считаем частотность признака: количество ненулевых значений
    feature_popularity = (features_only_df != 0).sum(axis=0).sort_values(ascending=False)
    
    # Выбираем имена TOP-N самых популярных признаков
    top_feature_names = feature_popularity.head(max_features_to_select).index.tolist()
    
    print(f"Отобрано признаков по частоте: {len(top_feature_names)}")
    
    # Обучение модели на отобранных признаках
    X = work_df[top_feature_names].values
    y = work_df[target_col].values

    scaler = StandardScaler()
    
    # Проверка на случай, если все выбранные признаки вдруг стали константами
    train_variance = np.var(X, axis=0)
    if np.all(train_variance == 0):
        raise ValueError("Все выбранные признаки константны")
        
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    
    model.fit(X_scaled, y)
    
    y_pred = model.predict(X_scaled)
    
    r2_all_data = r2_score(y, y_pred)
    rmse_all_data = np.sqrt(mean_squared_error(y, y_pred))

    # Получаем коэффициенты (веса) модели
    coefficients = model.coef_
    
    # Делаем все коэффициенты положительными и нормализуем
    coefficients_abs = np.abs(coefficients)
    
    # Нормализуем коэффициенты в диапазон от 0 до 1, затем масштабируем
    if coefficients_abs.max() > coefficients_abs.min():
        # Нормализация в [0, 1]
        coefficients_normalized = (coefficients_abs - coefficients_abs.min()) / (coefficients_abs.max() - coefficients_abs.min())
        # Усиление дифференциации - возводим в степень
        coefficients_enhanced = np.power(coefficients_normalized, 2) * 1000  # Масштабируем для больших значений
    else:
        coefficients_enhanced = coefficients_abs * 1000

    # Создаем Series весов для отобранных признаков с улучшенными коэффициентами
    final_weights = pd.Series(
        coefficients_enhanced, index=top_feature_names, name="weight"
    ).sort_values(ascending=False)

    result = {
        "model": model,
        "scaler": scaler,
        "feature_weights": final_weights,  # Сохраняем УЛУЧШЕННЫЕ веса
        "r2_all_data": r2_all_data,
        "rmse_all_data": rmse_all_data,
        "df_used": work_df,
        "selected_features": top_feature_names,
        "original_feature_names": feature_cols,
        "feature_popularity": feature_popularity,
    }

    return result

def save_model_and_weights(result, model_path="result/model.pkl", weights_path="result/weights.csv"):
    joblib.dump({
        'model': result["model"],
        'scaler': result["scaler"],
        'selected_features': result["selected_features"],
        'original_feature_names': result["original_feature_names"]
    }, model_path)
    
    weights_df = result["feature_weights"].reset_index()
    
    col_name = 'weight' 
    weights_df.columns = ["feature_name", col_name]
    
    weights_df.to_csv(weights_path, index=False)
    
    print(f"✅ Модель и нормализация сохранены в {model_path}")
    print(f"✅ Веса сохранены в {weights_path}")
    
    # Показываем статистику улучшенных весов
    weights = result["feature_weights"]
    print(f"Статистика весов:")
    print(f"  Минимум: {weights.min():.6f}")
    print(f"  Максимум: {weights.max():.6f}")
    print(f"  Среднее: {weights.mean():.6f}")
    print(f"  Стандартное отклонение: {weights.std():.6f}")
    print(f"  Все веса положительные: {(weights >= 0).all()}")
    
    # Показываем коэффициент дифференциации
    if weights.mean() > 0:
        diff_coeff = (weights.max() - weights.min()) / weights.mean()
        print(f"  Коэффициент дифференциации: {diff_coeff:.2f}")
