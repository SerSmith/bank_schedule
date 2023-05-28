"""Константы проекта
"""
import os

# get project root path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
        )
    )
RAW_DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'processed')
INTERIM_DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'interim')
EXTERNAL_DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'external')

TRAIN_DATA_FILEPATH = os.path.join(INTERIM_DATA_FOLDER, 'train.csv')

MODELS_FOLDER = os.path.join(PROJECT_ROOT, 'models')
LGBM_FOLDER = os.path.join(MODELS_FOLDER, 'lgbm')

# центр Москвы
CENTER_LAT = 55.755864
CENTER_LON = 37.617698

# Радиус Земли на широте Москвы
EARTH_R = 6363568

# random state
RS = 27

if __name__ == '__main__':
    print(PROJECT_ROOT)
    print(RAW_DATA_FOLDER)
