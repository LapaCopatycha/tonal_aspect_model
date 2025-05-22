# Модель аспектного анализа тональности отзывов
Данная модель выделяет и делает выводы относительно удовлетворённости покупателей определёнными характеристиками смартфонов, такие как: качество камеры, время работы от одной зарядки аккумулятора (автономность), цена/качество, общая работспособность и др.
Также данная модель помещена в docker контейнер и может использоваться, как api.

## Запуск программы
### Для обучения модели
1. Скачать данный репозиторий
2. Установить виртуальное окружение
3. Установить пакет pytorch
   ```python
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```
4. Установить все оставшиеся зависимости:
   ```python
   pip install -r dev-requirements.txt
   ```
5. Скачать модель эмбеддингов по [ссылке](https://github.com/LapaCopatycha/tonal_aspect_model/releases/download/v1.0.0-alpha/navec_hudlit_v1_12B_500K_300d_100q.tar) и поместить ее в директорию: `/taa_model/emb/`
6. Скачать размеченные данные для обучения по [ссылке](https://github.com/LapaCopatycha/tonal_aspect_model/releases/download/v1.0.0-alpha/train_reviews.csv) и тестирования по [ссылке](https://github.com/LapaCopatycha/tonal_aspect_model/releases/download/v1.0.0-alpha/test_reviews.csv) и поместить их в директорию: `/taa_model/data/`
7. Подготовить данные используя jupyter notebook eda, скачать его можно по [ссылке](https://github.com/LapaCopatycha/tonal_aspect_model/releases/download/v1.0.0-alpha/eda.ipynb) и поместить в директорию: `/taa_model/model/notebooks/`
8. Обучить модель используя jupyter notebook model fit, скачать его можно по [ссылке](https://github.com/LapaCopatycha/tonal_aspect_model/releases/download/v1.0.0-alpha/model_fit.ipynb) и поместить в директорию: `/taa_model/model/notebooks/`
9. При желании можете эксперементировать с моделями, с обработкой данных. As you wish.

### Для использования
Для использования модели, как docker контейнер нужно выполнить следующие действия.
1. Скачать данный репозиторий
2. Скачать модель эмбеддингов по [ссылке](https://github.com/LapaCopatycha/tonal_aspect_model/releases/download/v1.0.0-alpha/navec_hudlit_v1_12B_500K_300d_100q.tar) и поместить ее в директорию: `/taa_model/emb/`
3. Обучить самостоятельно модель или скачать её по [ссылке](https://github.com/LapaCopatycha/tonal_aspect_model/releases/download/v1.0.0-alpha/model_lstm_bidir.tar) и поместить в директорию: `/taa_model/model_history/`
4. Скачать название характеристик по [ссылке](https://github.com/LapaCopatycha/tonal_aspect_model/releases/download/v1.0.0-alpha/header.json) и поместить их в директорию: `/taa_model/data/`
5. Дальше требуется упаковать программу в docker image. Для этого в консоли, где находится проект нужно выполнить следующий код.
   ```
   docker buildx build -t taa-ml .
   ```
6. Далее нужно запустить контейнер из image, выполнив в консоли следующий код:
   ```
   docker run -it -p 80:80 taa-ml
   ```
7. Теперь можно отправлять запросы на разметку отзывов к api по адресу: `http://0.0.0.0:80/mark_review`


## Устройство модели
Модель состоит из:
1. Слоя векторного представления слов (эмбеддинг), используемая модель: [Наташа](https://natasha.github.io/).
2. Рекуретного слоя lstm
3. Линейного слоя

Модель обучена на около 1000 размеченных отзывов смартфонов с wildberries. Количество используемых характеристик: 14 = 7 положительных + 7 отрицательных.

## To-do list:
1. Заменить метрику модели на другую
2. Попробовать другие архитектуры ml моделей

## Дальнейший шаг использование модели
Построение реккомендаций на основе данной модели.
Для лучшего понимания идеи объясним ее на примере: Допустим у нас есть покупатель, которому очень важно чтобы смартфон обладал качественной камерой. Тогда мы ему будем предлагать те телефоны, в которых в отзывах другие покупатели отметили хорошее качество снимков и камеры. 
