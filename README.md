# МОДЕЛЬ ПО ПОСТРОЕНИЮ ОПТИМАЛЬНЫХ МАРШРУТОВ ИНКАССАЦИИ ПЛАТЕЖНЫХ ТЕРМИНАЛОВ.
# (ЗАДАЧА №6)
# Решение от команды Optimists

Репозиторий содержит код для решения задачи №6 в рамках конкурса "Лидеры цифровой трансформации 2023"
[https://leaders2022.innoagency.ru/task10](https://lk.leaders2023.innoagency.ru/)

## ДОКУМЕНТАЦИЯ
Подробности работы алгоритмов описаны по ссылке https://docs.google.com/document/d/19ftHeXtqAmW5It6GZB7rXeG6sPUV7aQRcBvD6-6h00k/edit?usp=sharing

## СТРУКТУРА
- **main.ipynb** - Ноутбук с примерами запуска
- **Result** - Содержит решения, на основе разных подходов
  - Пример 1
  - Пример 2
-  **src** - Кодовая база
-  **notebooks** - Дополнительные ноутбуки - исследования
-  **models** - Обученные прогнозные модели


## РАБОТА С МОДУЛЕМ

Модуль протестирован для python 3.10.6

Предполагается, что вы используете для контроля версий pyenv с плагином pyenv-virtualenv
Если у вас windows + pyenv, пропускаем шаг 2 и 5, на шаге 6 в папке с проектом (!)
делаем pyenv local 3.10.6
    
    0. Прочтите описание алгоритмов https://docs.google.com/document/d/19ftHeXtqAmW5It6GZB7rXeG6sPUV7aQRcBvD6-6h00k/edit?usp=sharing , а также примеры вызовов https://github.com/SerSmith/bank_schedule/blob/main/main.ipynb
    1. Установите python 3.10.6
       pyenv install 3.10.6
    2. Создайте виртуальное окружение 
       pyenv virtualenv 3.10.6 lct
    3. Клонируйте репозиторий git clone git@github.com:SerSmith/bank_schedule.git
    4. cd bank_schedule
    5. Активируйте виртуальное окружение
       pyenv activate lct
    6. pip install pip --upgrade
    7. Установите requirements
       pip install -r requirements.txt
    8. Установите репозиторий в виде библиотеки
       pip install -e ./.
    9. Что бы запустить оптимизацию на основе MILP вам понадобиться солвер. Инструкция по его установке:
https://github.com/coin-or/Cbc Раздел download(зависит от вашей OS).
    10. Добавте установленный солвер в ваш syspath
    11. Добавьте данные в /data/raw , там должно оказать 3 файла:
      11.1 Params.xlsx - набор основных констант(пример лежит в bank_schedule/data/raw, без необходимости просим не менять технические константы)
      11.2 terminal_data_hackathon v4.xlsx - Как в задании
      11.3 times v4.csv - Как в задании
   








 
