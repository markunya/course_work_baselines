# Фреймворк для задачи улучшения речи

В данном репозитории представлен код для обучения и экспериментов с рядом современных генеративно-состязательных моделей (GAN) в задачах улучшения качества речи. Фреймворк ориентирован на гибкость, модульность и воспроизводимость экспериментов. Поддерживается оценка качества с использованием актуальных метрик.

## Настройка окружения
```bash
git clone https://github.com/markunya/course_work_baselines.git
cd course_work_baselines
conda create -n myenv python=3.12.8 ffmpeg
source activate myenv
pip install -r requirements.txt
```

## Данные и датасеты

## Модели

## Метрики

## Аугментации

## Конфигурация

## Обучение и инференс
Пример запуска обучения:
```bash
python train.py exp.config_path=config.yaml exp.run_name=train
```

Пример запуска инференса:
```bash
python inference.py exp.config_path=config.yaml exp.run_name=inf
```
