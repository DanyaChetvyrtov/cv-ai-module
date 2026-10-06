# CV Face Service

Локальный Python-модуль для проверки сотрудников по лицу. Модуль больше не выполняет
универсальную YOLO-детекцию объектов: его единственная CV-задача — получить стабильный
face embedding, который Kotlin BFF использует для регистрации и поиска сотрудников.

## Как это работает

1. FastAPI принимает фотографию через `POST /faces/embedding`.
2. YuNet находит ровно одно лицо и его landmarks.
3. SFace выравнивает лицо и строит нормализованный 128-мерный embedding.
4. Python возвращает embedding, bounding box лица и идентификатор версии модели.
5. Python не хранит сотрудников, фотографии и Keycloak-данные. Сопоставление embedding
   с базой сотрудников выполняет Kotlin-модуль.

Обе модели работают локально на CPU. Собственную модель для добавления нового сотрудника
обучать не требуется.

## Запуск

Нужен Python 3.11+ (рекомендуется 3.12):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
uvicorn cv_test.api:create_app --factory --host 127.0.0.1 --port 8000
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn cv_test.api:create_app --factory --host 127.0.0.1 --port 8000
```

При первом запуске pinned-модели YuNet и SFace скачиваются из OpenCV Zoo в `models/`,
проверяются по SHA-256 и после этого могут использоваться офлайн.

## HTTP API

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/faces/embedding -F "image=@person.jpg"
```

Успешный ответ:

```json
{
  "model": "sface-2021dec:...",
  "embedding": [0.12, -0.04, 0.08],
  "bbox": {"x1": 120.0, "y1": 70.0, "x2": 360.0, "y2": 410.0},
  "inference_ms": 18.4
}
```

В реальном ответе `embedding` содержит ровно 128 чисел. Требуется одно достаточно
крупное лицо. Фото без лица, с несколькими лицами или слишком маленьким лицом получает
HTTP 422. Поддерживаются JPEG, PNG и WEBP.

## Настройки

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `CV_MAX_UPLOAD_BYTES` | `10485760` | Максимальный размер файла |
| `CV_MAX_IMAGE_PIXELS` | `20000000` | Максимальное число пикселей |
| `CV_FACE_ENABLED` | `true` | Включить face inference |
| `CV_FACE_MODELS_PATH` | `models` | Каталог YuNet/SFace |
| `CV_MIN_FACE_SIZE` | `60` | Минимальная сторона лица в пикселях |

## Интеграция

В общем приложении React не обращается к этому сервису напрямую:

```text
React -> Kotlin BFF -> Python CV
                    -> PostgreSQL
```

Kotlin отправляет фотографию на `/faces/embedding`, получает embedding и сам решает,
является ли лицо уже зарегистрированным сотрудником. CV-сервис не принимает решение
ALLOW/DENY и не хранит бизнес-данные.

## Проверка

```bash
ruff check .
ruff format --check .
pytest -m "not integration"
```

Интеграционный тест настоящих YuNet/SFace:

```bash
python scripts/fetch_test_face.py /tmp/cv-test-face.png
CV_RUN_FACE_INTEGRATION=1 CV_FACE_TEST_IMAGE=/tmp/cv-test-face.png \
  pytest tests/test_face_integration.py
```

Исходные фотографии не сохраняются и не логируются. В учебной реализации нет
liveness/anti-spoofing, поэтому результат распознавания сам по себе не должен открывать
реальный турникет.

YuNet распространяется по MIT, SFace — Apache-2.0; notices находятся в `licenses/`.
