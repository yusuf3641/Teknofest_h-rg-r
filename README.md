# HürGör Edge AI pipeline

TEKNOFEST 2026 Havacılıkta Yapay Zeka için kapalı yerel ağda çalışan, bounded producer-consumer haberleşme ve görüntü işleme iskeletidir.

Resmî sıra kuralı korunur:

`GET kare -> inference -> POST aynı kare -> ACK -> sonraki GET`

## Mimari

Üç bağımsız thread vardır:

1. `Producer-Network-IN`: Kendi `asyncio` event loop'u ve HTTP istemcisiyle frame metadata/görselini alır, `Input Queue` içine koyar.
2. `Worker-AI-Engine`: Input Queue'dan alır; nesne tespiti, pozisyon kestirimi ve referans eşlemeyi çalıştırıp `Output Queue` içine koyar.
3. `Consumer-Network-OUT`: Pydantic ile son validasyonu yapar ve sonucu POST eder.

Input ve Output Queue `queue.Queue(maxsize=3)` kullanır. Producer, Consumer'dan doğru frame ACK'i gelmeden yeni GET yapmaz. Böylece thread'ler ayrıdır ancak şartnamedeki “POST olmadan sonraki frame alınamaz” kuralı ihlal edilmez.

Tüm ağ ve inference süreleri milisaniye cinsinden hem konsola hem dönen `system.log` dosyasına yazılır. `print()` kullanılmaz.

## Kurulum

Python 3.11 veya üzeri gerekir:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,ai]'
cp .env.example .env
pytest
```

## Mock sunucu

Sentetik frame'lerle:

```bash
hurgor-mock-server --port 5000
```

Yerel videoyu sırayla frame'lere bölerek:

```bash
hurgor-mock-server --video /absolute/path/test.mp4 --port 5000
```

Durum ve API dokümanı:

- `http://127.0.0.1:5000/api/status`
- `http://127.0.0.1:5000/docs`

Fault injection ayarları:

- `HURGOR_MOCK_CORRUPT_EVERY=10`
- `HURGOR_MOCK_EMPTY_EVERY=15`
- `HURGOR_MOCK_GET_DELAY_MS=...`
- `HURGOR_MOCK_POST_DELAY_MS=...`

## İstemci

`.env` adresiyle:

```bash
hurgor-client
```

Yarışma günü dinamik IP/port ile:

```bash
hurgor-client --server-ip 127.0.0.25 --port 5000
```

Kısa koşu:

```bash
hurgor-client --max-frames 100
```

Şartnamenin 25-27. sayfalarındaki sözleşmeye göre:

- GET cevabı tek elemanlı JSON listesidir ve mock sunucu `health_status` üretir.
- İstemci, dokümandaki metin/şekil farkı nedeniyle hem `health_status` hem
  `gps_health_status` okuyabilir.
- POST gövdesi her zaman tek elemanlı JSON listesidir.
- `user`, `HURGOR_USER_URL` değerinden mutlak URL olarak POST'a eklenir.
- `session`, `HURGOR_SESSION_URL` ile mock GET cevabında üretilir ve gelen oturumla
  karşılaştırılır; resmî POST örneğinde `session` alanı bulunmadığı için POST'a eklenmez.
- Nesne sınıfları `http://SERVER/classes/CLASS_ID/` biçiminde gönderilir.
- Tahmin kimliği deterministik SHA-256 özetinden, JSON için güvenli tamsayı aralığında üretilir.

## Model runtime ve export

Runtime `.pt`, `.pth` veya `.h5` yüklemez. YOLO modeli ONNX'e çevrilir:

```bash
python -m pip install -e '.[export]'
hurgor-export-yolo weights/best.pt --target onnx --image-size 640
```

Sonra `.env` içinde:

```dotenv
HURGOR_YOLO_ONNX_PATH=/absolute/path/best.onnx
HURGOR_REFERENCE_IMAGES_DIR=/absolute/path/references
```

ONNX Runtime provider sırası `TensorRT -> CUDA -> CPU` şeklindedir. NVIDIA sistemde `TensorrtExecutionProvider` kuruluysa ONNX modelinden FP16 TensorRT engine cache otomatik oluşturulur. “3 kat hız” sabit bir garanti değildir; hedef donanımda benchmark ile doğrulanmalıdır.

AI modülleri:

- YOLOv8 uyumlu ONNX detector ve NMS
- Yoğun topolojik gürültü için çok eşikli, persistence-inspired connected-component ön filtresi
- UAP/UAİ ve engeller için ters izdüşüm tabanlı 3B frustum/AABB IoU analizi
- GPS sağlıksızken Lucas-Kanade optik akış ve `SE(3)` pose güncellemesi
- RGB/termal normalize edilmiş ORB özellikleri, RANSAC homografi ve `detected_undefined_objects`

Önemli sınırlar:

- Hafif topolojik filtre tam persistent-homology/TDA kütüphanesi değildir.
- ORB çapraz spektral eşlemenin baseline'ıdır; gerçek SuperGlue ONNX ağırlıkları geldiğinde `UndefinedObjectMatcher` adaptörüyle değiştirilmelidir.
- Monoküler görsel odometride mutlak ölçek doğrudan gözlenemez; mevcut baseline kamera yüksekliği ve iç parametrelerini kullanır.
- 3B analiz iç karardır; resmî JSON yalnızca 2B kutuları gönderir.

Model veya görüntü bozulursa boş tespit ve son güvenilir/0 konumla geçerli fallback JSON gönderilir. JSON'a `NaN` yazılmaz; standart JSON ve Pydantic doğrulaması finite sayı ister.

## Graceful degradation

Ardışık 5 inference 800 ms'yi aşarsa ağır topolojik ön işleme ve referans eşleme devre dışı bırakılır; hafif detector ve pozisyon matematiği devam eder. Ardışık 10 frame 250 ms altına indiğinde ağır modüller yeniden açılır. Eşikler `.env.example` üzerinden değiştirilebilir.

## Docker ve çevrimdışı kurulum

CPU container:

```bash
docker compose up --build
```

NVIDIA GPU için host driver, NVIDIA Container Toolkit ve GPU uyumlu ONNX Runtime/TensorRT image gerekir; CUDA driver container içine gömülmez.

Yarışma ağı internetsiz olduğu için hedef işletim sistemi, mimari ve Python sürümüyle aynı makinede wheelhouse hazırlanmalıdır:

```bash
python -m pip wheel -w wheelhouse '.[ai]'
python -m pip install --no-index --find-links wheelhouse hurgor-edge-pipeline
```

Endpoint ve son JSON şeması yarışma tarafından güncellendiğinde `config.py`, `models.py` ve API adaptörü birlikte güncellenmelidir.
