import geopandas as gpd
import osmnx as ox
import quackosm as qosm
import pyogrio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Polygon
import pandas as pd
import numpy as np
from functools import reduce
import json
from matplotlib.colors import LogNorm

def read_map(path_geojson: str, path_pbf: str):
    """
    Загружает границу региона и все слои из PBF-файла.

    Parameters
    ----------
    path_geojson : str
        Путь к GeoJSON-файлу с границей региона.
    path_pbf : str
        Путь к PBF-файлу с данными OSM.

    Returns
    -------
    tuple
        (boundary, layers, all_layers)
        - boundary : GeoDataFrame — граница региона
        - layers : ndarray — список названий слоёв
        - all_layers : dict — словарь с GeoDataFrame для каждого слоя
    """
    # Границы субъекта
    boundary = gpd.read_file(path_geojson)

    # Слои на карте
    layers = pyogrio.list_layers(path_pbf)[:, 0]

    # Создаём словарь для хранения GeoDataFrame
    all_layers = {}

    # Загружаем каждый слой
    for layer in layers:
        try:
            gdf = gpd.read_file(
                path_pbf,
                engine="pyogrio",
                layer=layer,
                use_arrow=True,
                on_invalid="ignore",
            )
            all_layers[layer] = gdf
        except Exception as e:
            print(f"Ошибка при загрузке {layer}: {e}")

    return boundary, layers, all_layers


def create_grid_within_boundary(
    boundary_gdf: gpd.GeoDataFrame, cell_size_km: float = 1
):
    """
    Создаёт сетку ячеек заданного размера строго внутри границы региона.

    Parameters
    ----------
    boundary_gdf : GeoDataFrame
        Геометрия границы региона.
    cell_size_km : float, default=1
        Размер ячейки в километрах.

    Returns
    -------
    tuple
        (grid_filtered, cell_size_deg_lon, cell_size_deg_lat)
        - grid_filtered : GeoDataFrame — сетка ячеек внутри границы
        - cell_size_deg_lon : float — ширина ячейки в градусах
        - cell_size_deg_lat : float — высота ячейки в градусах
    """
    minx, miny, maxx, maxy = boundary_gdf.total_bounds

    center_lat = (miny + maxy) / 2
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat))

    cell_size_deg_lat = cell_size_km / km_per_deg_lat
    cell_size_deg_lon = cell_size_km / km_per_deg_lon

    n_cells_x = int(np.ceil((maxx - minx) / cell_size_deg_lon))
    n_cells_y = int(np.ceil((maxy - miny) / cell_size_deg_lat))

    print(f"Размер ячейки: {cell_size_km} км")
    print(
        f"Сетка в bounding box: {n_cells_x} × {n_cells_y} = {n_cells_x * n_cells_y} ячеек"
    )

    all_cells = []
    for i in range(n_cells_x):
        for j in range(n_cells_y):
            x_left = minx + i * cell_size_deg_lon
            x_right = x_left + cell_size_deg_lon
            y_bottom = miny + j * cell_size_deg_lat
            y_top = y_bottom + cell_size_deg_lat

            cell = Polygon(
                [
                    (x_left, y_bottom),
                    (x_right, y_bottom),
                    (x_right, y_top),
                    (x_left, y_top),
                ]
            )

            all_cells.append(
                {"cell_id": f"{i}_{j}", "cell_x": i, "cell_y": j, "geometry": cell}
            )

    grid_full = gpd.GeoDataFrame(all_cells, crs="EPSG:4326")
    mask = grid_full.intersects(boundary_gdf.unary_union)
    grid_filtered = grid_full[mask].copy()
    grid_filtered["geometry"] = grid_filtered["geometry"].intersection(
        boundary_gdf.unary_union
    )
    grid_filtered = grid_filtered[~grid_filtered.is_empty]
    grid_filtered = grid_filtered.reset_index(drop=True)

    print(f"Ячеек внутри границы: {len(grid_filtered)}")
    return grid_filtered, cell_size_deg_lon, cell_size_deg_lat


def get_object_type(row: pd.Series, layer_name: str) -> str:
    """
    Определяет тип объекта на основе OSM-тегов.

    Parameters
    ----------
    row : pd.Series
        Строка с атрибутами объекта OSM.
    layer_name : str
        Название слоя (points, lines, multipolygons и т.д.)

    Returns
    -------
    str
        Строка с типом объекта в формате "тег:значение" или "слой:other".
    """
    # Приоритетные теги для разных слоёв
    priority_tags = [
        "building",
        "highway",
        "amenity",
        "landuse",
        "leisure",
        "natural",
        "waterway",
        "railway",
        "shop",
        "tourism",
        "historic",
        "man_made",
        "power",
        "barrier",
    ]

    # Сначала ищем осмысленные теги
    for tag in priority_tags:
        if tag in row and pd.notna(row[tag]) and row[tag] != "":
            tag_value = str(row[tag])
            if tag_value.lower() == "yes":
                return f"{tag}:[не указан]"
            return f"{tag}:{tag_value}"

    # Если есть колонка 'type' (для отношений)
    if "type" in row and pd.notna(row["type"]) and row["type"] != "":
        return f"type:{row['type']}"

    # Если есть колонка 'geometry_type'
    if "geometry_type" in row and pd.notna(row["geometry_type"]):
        return f"geom:{row['geometry_type']}"

    # Универсальное определение по названию слоя
    base_type = _extract_base_type(layer_name)
    return f"{base_type}:other"


def _extract_base_type(layer_name: str) -> str:
    """
    Извлекает базовый тип из названия слоя (вспомогательная функция).

    Parameters
    ----------
    layer_name : str
        Название слоя (например, 'points', 'multilinestrings').

    Returns
    -------
    str
        Базовый тип слоя (например, 'point', 'multiline').
    """
    # Специальные случаи
    special_cases = {"multilinestrings": "multiline", "other_relations": "relation"}

    if layer_name in special_cases:
        return special_cases[layer_name]

    # Убираем окончание 's' для множественного числа
    base = layer_name.rstrip("s")

    return base


def aggregate_layer_to_grid(
    layer_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
    layer_name: str,
    boundary_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Агрегирует данные из слоя в сетку с сохранением информации о типах объектов.

    Parameters
    ----------
    layer_gdf : GeoDataFrame
        Данные слоя для агрегации.
    grid_gdf : GeoDataFrame
        Сетка ячеек.
    layer_name : str
        Название слоя.
    boundary_gdf : GeoDataFrame
        Граница региона (используется для пространственного фильтра).

    Returns
    -------
    pd.DataFrame
        DataFrame с колонками: cell_id, count_{layer_name}, types_{layer_name}
    """
    joined = gpd.sjoin(layer_gdf, grid_gdf, how="inner", predicate="within")

    # Базовая агрегация: количество объектов по ячейкам
    basic_agg = joined.groupby("cell_id").size().reset_index(name=f"count_{layer_name}")

    # Определяем тип каждого объекта
    if len(joined) > 0:
        joined["object_type"] = joined.apply(
            lambda row: get_object_type(row, layer_name), axis=1
        )

        # Группируем по ячейке и типу
        type_agg = (
            joined.groupby(["cell_id", "object_type"]).size().reset_index(name="count")
        )

        # Превращаем в словарь {cell_id: {type1: count1, type2: count2}}
        type_dict = {}
        for cell_id, group in type_agg.groupby("cell_id"):
            type_dict[cell_id] = dict(zip(group["object_type"], group["count"]))

        # Добавляем как отдельную колонку
        basic_agg[f"types_{layer_name}"] = (
            basic_agg["cell_id"].map(type_dict).fillna({})
        )
    else:
        basic_agg[f"types_{layer_name}"] = [{} for _ in range(len(basic_agg))]

    return basic_agg


def create_cell_description(row: pd.Series, all_layers: dict) -> tuple:
    """
    Создаёт описание ячейки с детализацией по типам объектов.

    Parameters
    ----------
    row : pd.Series
        Строка со статистикой по ячейке.
    all_layers : dict
        Словарь со всеми слоями (нужен для получения названий).

    Returns
    -------
    tuple
        (description, types_json)
        - description : str — текстовое описание для людей
        - types_json : str — JSON со структурой для машинного чтения
    """
    total_objects = 0
    layer_details = []
    all_types = {}

    for layer_name in all_layers.keys():
        count_col = f"count_{layer_name}"
        types_col = f"types_{layer_name}"

        if count_col in row and row[count_col] > 0:
            count = int(row[count_col])
            total_objects += count

            # Получаем типы для этого слоя
            types_dict = row.get(types_col, {})
            if isinstance(types_dict, dict) and types_dict:
                # Сортируем по убыванию и берём топ-3 типа
                sorted_types = sorted(
                    types_dict.items(), key=lambda x: x[1], reverse=True
                )[:3]
                types_str = ", ".join([f"{t}({c})" for t, c in sorted_types])
                layer_details.append(f"{layer_name}({count}) [{types_str}]")
                # Добавляем в общий словарь типов
                for t, c in types_dict.items():
                    all_types[t] = all_types.get(t, 0) + c
            else:
                layer_details.append(f"{layer_name}({count})")

    if total_objects == 0:
        return "Пустая ячейка (нет данных OSM)", "{}"

    # Определяем доминирующий слой
    layer_counts = {
        layer: int(row.get(f"count_{layer}", 0)) for layer in all_layers.keys()
    }
    dominant_layer = max(layer_counts, key=layer_counts.get)

    # Определяем доминирующий тип (если есть)
    dominant_type = ""
    if all_types:
        dominant_type_obj = max(all_types.items(), key=lambda x: x[1])
        dominant_type = (
            f", доминирующий тип: {dominant_type_obj[0]}({dominant_type_obj[1]})"
        )

    # Создаём описание
    description = (
        f"Объектов: {total_objects}. Преобладают: {dominant_layer}{dominant_type}. "
    )
    description += "Состав: " + "; ".join(layer_details[:5])

    if len(layer_details) > 5:
        description += f" и ещё {len(layer_details)-5} слоёв"

    # Создаём JSON с полной структурой
    full_types_json = {
        "total": total_objects,
        "dominant_layer": dominant_layer,
        "layers": {
            layer: {
                "count": int(row.get(f"count_{layer}", 0)),
                "types": row.get(f"types_{layer}", {}),
            }
            for layer in all_layers.keys()
            if row.get(f"count_{layer}", 0) > 0
        },
    }

    return description, json.dumps(full_types_json, ensure_ascii=False)


def safe_json_parse(x):
    """
    Безопасно парсит JSON-строку.

    Parameters
    ----------
    x : any
        Входное значение (строка, NaN, None и т.д.)

    Returns
    -------
    dict
        Распарсенный JSON или пустой словарь в случае ошибки.
    """
    if pd.isna(x) or x == "" or x == "{}":
        return {}
    try:
        return json.loads(x)
    except:
        return {"error": "invalid json", "raw": str(x)[:100]}


def aggregate_all_layers_to_grid(
    all_layers: dict, grid_gdf: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Агрегирует все слои в единую таблицу.

    Parameters
    ----------
    all_layers : dict
        Словарь со всеми слоями (GeoDataFrame).
    grid_gdf : GeoDataFrame
        Сетка ячеек.
    boundary : GeoDataFrame
        Граница региона.

    Returns
    -------
    pd.DataFrame
        Объединённая статистика по всем ячейкам и слоям.
    """
    aggregated_data = []

    for layer_name, gdf in all_layers.items():

        gdf_to_use = gdf

        agg = aggregate_layer_to_grid(gdf_to_use, grid_gdf, layer_name, boundary)
        aggregated_data.append(agg)


    final_stats_numeric = reduce(
        lambda left, right: pd.merge(left, right, on="cell_id", how="outer"),
        aggregated_data,
    ).fillna(0)

    final_stats = final_stats_numeric.merge(
        grid_gdf[["cell_id", "cell_x", "cell_y", "geometry"]],
        on="cell_id",
        how="right",
    ).fillna(0)

    print(f"\nИтоговая таблица: {len(final_stats)} ячеек")
    print(f"Колонки: {list(final_stats.columns)}")

    return final_stats