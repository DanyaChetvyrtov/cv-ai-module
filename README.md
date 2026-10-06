# CV Test

Небольшой рабочий Python-проект для **детекции объектов на изображении**.
Готовая модель **YOLO11n** распознаёт 80 классов COCO: людей, автомобили,
велосипеды, животных и другие объекты. Обучать собственную модель для запуска не нужно.

Доступны три способа использования: HTTP API на FastAPI, команда `cv-detect`
и прямой вызов `YoloDetector` из Python. OpenCV используется для представления
изображений и рисования рамок, PyTorch — для вычислений модели.

## Быстрый запуск

Нужен Python 3.11+; рекомендуемый вариант — Python 3.12. GPU не требуется.

```bash
git clone https://github.com/DanyaChetvyrtov/cv-ai-module.git
cd cv-test
python -m venv .venv
```

Активировать окружение:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Установить CPU-версию PyTorch и проект:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
uvicorn cv_test.api:create_app --factory --host 127.0.0.1 --port 8000
```

При первом запуске Ultralytics скачивает `yolo11n.pt` в `models/` — нужен интернет.
Модель загружается один раз, выполняется прогрев, затем сервер начинает принимать
запросы. Следующие запуски используют сохранённые веса. Можно заранее положить
доверенный файл весов по пути `CV_MODEL_PATH` для запуска без сети.

Открыть **http://localhost:8000/docs**, выбрать `POST /vision/detect`, нажать
`Try it out`, загрузить фотографию и выполнить запрос.

## HTTP API

```bash
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/vision/detect?confidence=0.25" -F "image=@photo.jpg"
```

Поле multipart называется **`image`**. `confidence` — необязательный query-параметр
от `0.01` до `1.0`. Поддерживаются JPEG, PNG и WEBP.

Пример структуры ответа (числа иллюстративные):

```json
{
  "model": "yolo11n.pt",
  "width": 1280,
  "height": 720,
  "inference_ms": 120.5,
  "detections": [
    {
      "class_id": 0,
      "label": "person",
      "confidence": 0.93,
      "bbox": {"x1": 110.0, "y1": 80.0, "x2": 360.0, "y2": 690.0}
    }
  ]
}
```

`bbox` содержит пиксельные координаты верхнего левого и нижнего правого углов.
Начало координат — верхний левый угол изображения. EXIF-ориентация применяется
до распознавания; `width`, `height` и рамки относятся к повёрнутому изображению.
`inference_ms` включает вызов модели и преобразование результата; загрузка файла,
декодирование и ожидание очереди сюда не входят. Если объекты не найдены,
возвращается HTTP 200 и пустой `detections`.

Ошибки: `400` — пустое/повреждённое изображение, `413` — превышен лимит,
`415` — неподдерживаемый формат, `422` — неверные параметры или отсутствует файл,
`503` — ошибка вычисления модели.

## Обработка фотографии без сервера

```bash
cv-detect photo.jpg --output outputs/result.jpg --json outputs/result.json
```

Сохраняет фотографию с рамками и JSON. Без `--json` результат печатается в консоль.
Для первого эксперимента можно взять демонстрационную фотографию Ultralytics:

```bash
curl -L https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg -o photo.jpg
cv-detect photo.jpg --output outputs/result.jpg --json outputs/result.json
```

## Использование как модуля

```python
from pathlib import Path

from cv_test.config import Settings
from cv_test.detector import YoloDetector
from cv_test.images import decode_image

settings = Settings()
detector = YoloDetector(settings)  # Создать один раз и переиспользовать.
image = decode_image(
    Path("photo.jpg").read_bytes(),
    max_bytes=settings.max_upload_bytes,
    max_pixels=settings.max_image_pixels,
)
result = detector.detect(image, confidence=0.25)
print(result.model_dump_json(indent=2))
```

## Интеграция с Kotlin / Spring

Spring отправляет multipart HTTP-запрос в этот сервис и получает обычный JSON.
Python и модель запускаются отдельным процессом или контейнером.
Пример клиента на `RestClient` — [examples/SpringVisionClient.kt](examples/SpringVisionClient.kt).
Для Docker-сети адрес сервиса — `http://cv:8000`, для локального запуска —
`http://localhost:8000`. В клиенте стоит задать таймауты под время CPU-распознавания.

## Docker

Dockerfile этого модуля собирает CPU-образ API. Запуск всего приложения и хранение весов
настраиваются единственным Compose-файлом в корневом репозитории
[cv-complex-test](https://github.com/DanyaChetvyrtov/cv-root#запуск-всего-проекта).
В этом модуле Compose-файлов нет. Команды Docker Compose выполняйте из корня общего проекта.

В общем стенде этот сервис доступен только внутри Docker-сети по `http://cv:8000`.
React обращается к Kotlin BFF, а BFF передаёт сюда изображение. Порт Python на хост не публикуется.
Для детекции через приложение нужны BFF-сессия, роль USER и CSRF.
Веса сохраняются в volume `cv-test_model-cache`; первый старт требует интернета.
Для отдельной разработки API на `127.0.0.1:8000` используйте локальный запуск выше.

## Настройки

Можно скопировать `.env.example` в `.env` либо задать переменные окружения.

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `CV_MODEL_PATH` | `models/yolo11n.pt` | Имя/путь к весам детекции |
| `CV_DEVICE` | `cpu` | `cpu`, `0` для CUDA GPU или `mps` для Apple GPU |
| `CV_IMAGE_SIZE` | `640` | Размер входа модели, кратный 32, от 32 до 2048 |
| `CV_CONFIDENCE` | `0.25` | Минимальная уверенность детекции |
| `CV_MAX_UPLOAD_BYTES` | `10485760` | Максимальный размер файла, 10 MiB |
| `CV_MAX_IMAGE_PIXELS` | `20000000` | Максимальное число пикселей |

Для CUDA требуется соответствующая сборка PyTorch вместо CPU-сборки из инструкции.
Стандартный контейнер рассчитан на CPU.

## Проверка

```bash
ruff check .
ruff format --check .
pytest -m "not integration"
```

Обычные тесты проверяют API, декодирование и преобразование результата модели;
им не нужны веса или сеть. Отдельный интеграционный тест запускает настоящую
YOLO11n на встроенной фотографии автобуса, проверяет HTTP API и CLI:

```bash
# Linux / macOS; при первом запуске скачиваются веса
CV_RUN_INTEGRATION=1 pytest -m integration

# Windows PowerShell
$env:CV_RUN_INTEGRATION="1"
pytest -m integration
```

## Что учитывать при дальнейшем развитии

Модель распознаёт классы COCO. Для специфичных объектов проекта понадобится
подходящая готовая модель или дообучение на размеченных изображениях. Здесь
реализована обработка одного изображения за запрос. Одновременные вычисления
одной модели сериализуются; для этого небольшого сервиса используйте один worker.
Разделение на `detector.py`, `images.py` и `api.py` позволяет заменить модель,
сохранив HTTP-контракт.

Документация используемых библиотек:
[YOLO11](https://docs.ultralytics.com/models/yolo11/),
[Ultralytics Predict](https://docs.ultralytics.com/modes/predict/),
[FastAPI UploadFile](https://fastapi.tiangolo.com/tutorial/request-files/).
У Ultralytics есть условия AGPL-3.0 и коммерческого лицензирования:
[официальная информация](https://www.ultralytics.com/license).

## Local face embeddings for the employee registry

`POST /faces/embedding` accepts multipart `image`. YuNet detects a single face and its five landmarks;
SFace aligns/crops the face and extracts a normalized 128-dimensional embedding. Both models run locally on CPU.
The internal response contains `model` (including the SFace SHA-256), `embedding`, `bbox`, and `inference_ms`.
The BFF stores/templates and searches employees; the CV service has no employee database or Keycloak credentials.
This endpoint remains private in the root Docker network and is never proxied directly to React.

Exactly one face is required, with a detection score of at least 0.9 and a minimum side of 60 pixels after downscaling
large photos to a 1280-pixel long side. Invalid/no/multiple/small faces return 422. Malformed input is 400, unsupported
formats 415, oversized input 413. Inference failure is 503. No photos are persisted and the vector is never logged.
Inference is serialized because the OpenCV DNN models keep mutable state.

`CV_FACE_MODELS_PATH` defaults to `models`. At startup the two pinned OpenCV Zoo ONNX models are downloaded once,
installed atomically and checked by SHA-256. Valid cached files work offline; a corrupt cache is repaired.
`CV_FACE_ENABLED=false` disables this feature explicitly (the endpoint then returns 503).
YuNet uses the MIT license, SFace Apache-2.0; original notices are in `licenses/` and included in the Docker image.
Model source revision: `47534e27c9851bb1128ccc0102f1145e27f23f98`.

* [OpenCV recognition tutorial](https://docs.opencv.org/4.11.0/d0/dd4/tutorial_dnn_face.html)
* [YuNet model](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet)
* [SFace model](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface)

The employee demo has no liveness/anti-spoofing: a photo displayed to a camera can match. It is a photo comparison
prototype, and the thresholds need calibration on your data before any access-control use.

Real face integration test (independent of YOLO inference):

```bash
python scripts/fetch_test_face.py /tmp/cv-test-face.png
CV_RUN_FACE_INTEGRATION=1 CV_FACE_TEST_IMAGE=/tmp/cv-test-face.png pytest tests/test_face_integration.py
```

The checksum-pinned fixture comes from scikit-image's public-domain NASA astronaut photo;
[provenance](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.astronaut).
CI tests actual YuNet/SFace, a resized query, no face and multiple faces.
