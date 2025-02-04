# Воспроизведение бейзлайнов

## Модели

#### HiFiGAN
Ссылка на оригинальную статью: https://arxiv.org/abs/2010.05646

Файл конфигурации: configs/hifigan_config.yaml

Процесс обучения и логи: https://wandb.ai/bondarenko_mark-higher-school-of-economics/hifigan

Чекпоинты: 

#### HiFi++
Ссылка на оригинальную статью: https://arxiv.org/abs/2203.13086

Файлы конфиграции: configs/hifi++_bwe1_config.yaml, configs/hifi++_bwe2_config.yaml, configs/hifi++_bwe4_config.yaml, configs/hifi++_denoise_config.yaml

Процесс обучения и логи: https://wandb.ai/bondarenko_mark-higher-school-of-economics/hifi++

Чекпоинты:

#### Finally
Ссылка на оригинальную статью: https://arxiv.org/abs/2410.05920

Файлы конфигурации: configs/finally_stage1_config.yaml, configs/finally_stage2_config.yaml, configs/finally_stage3_config.yaml

Процесс обучения и логи: -

Чекпоинты: -

## Конфигурация

Для обучения и инференса в файле конфигурации нужно выбрать тренера. На данный момент написаны hifigan_trainer и hifi++_trainer для обучения соответствующих моделей (finally в разработке). Указать датасет, модели и соответствующие каждой функции потерь с коэффициентами в сумме, оптимизаторы и шедуллеры. По желанию можно настроить шаг валидации, сохранения чекпоинта, количество логируемых записей, а также валидациооные метрики и метрики для инференса. Можно включить логирование wandb. Доступна настройка параметров вычисления мел-спектрогрммы и других значений необходимых для обработки аудио. В файлах конфиграции поддерживается наследование через ключевое слово inherit. 

Пример запуска обучения:
```bash
python train.py exp.config_path=configs/hifigan_config.yaml exp.run_name=train
```

Пример запуска инференса:
```bash
python inference.py exp.config_path=configs/hifigan_config.yaml exp.run_name=inf
```

## Датасеты и данные
Сплиты, разделяющие данные на тренировочные и валидационные наборы, доступны в datasets/splits.

-VCTK. Датасет VCTK содержит высококачественные записи чистой речи, выполненные множеством дикторов. В соответствии с оригинальными статьями, этот датасет использовался для обучения моделей HiFiGAN и HiFi++ в задаче ширины полосы. Частота дискретизации была снижена с 48 кГц до 24 кГц для HiFiGAN и до 16 кГц для HiFi++ с помощью скрипта resample\_dataset.py (см. раздел скрипты).

-VoiceBank-Demand. Датасет VoiceBank-Demand включает записи чистой речи и их зашумленные версии. Он использовался для обучения модели HiFi++ в задаче удаления шума. Перед обучением папки с чистыми записями для тестирования и обучения были объединены в одну, аналогично было сделано с зашумленными записями. Не было необходимости сохранять две отдельные папки, так как существуют файлы сплитов.

В контексте конфигурации для HiFiGAN использовался mel_dataset, для bwe HiFi++ - vctk, а для denoise HiFi++ - voicebank.

## Шедулеры и оптимизаторы
Доступны оптимизаторы sgd, adam и adamW, а также шедулеры exponential и multi_step, соответствующие аналагам из pytorch. Дополнительно поддерживается warmup и reduce_time. reduce_time отвечает за период шага шедулера, доступно step, epoch и period. Позволяющие делать шаг каждую итерацию, эпоху или кастамное число итераций.

## Метрики
Доступны метрики mosnet, wb_pesq, stoi, l1_mel_dif и si_sdr. Для метрики mosnet нужны предобученные веса в папке metrics/weights. Их можно загрузить с помощью скрипта download_extract_weigths.sh из директории metrics.

```bash
chmod +x download_extract_weights.sh
./download_extract_weights.sh
```

## Настройка окружения
```bash
git clone https://github.com/markunya/course_work_baselines.git
cd course_work_baselines
conda create -n myenv python=3.12.8 ffmpeg
source activate myenv
pip install -r requirements.txt
```

## Скрипты
trainval_split.py - нужен для создания файлов сплита датасета в формате txt. Пример:
```bash
python trainval_split.py --dataset_dir DATASET_DIR --output_dir OUTPUT_DIR --val_size 0.1
```

resample_dataset.py - создает новый датасет на основе старого с другой частотой дискретизации. Сохраняет иерархию. Пример:
```bash
python resample_dataset.py --dataset_dir DATASET_DIR --output_dir OUTPUT_DIR --target_sr 16000
```
