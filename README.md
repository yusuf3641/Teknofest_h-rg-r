# HürGör Edge AI pipeline

TEKNOFEST 2026 Havacılıkta Yapay Zeka için kapalı yerel ağda çalışan, bounded producer-consumer haberleşme ve görüntü işleme iskeletidir.

Resmî sıra kuralı korunur:

`GET kare -> inference -> POST aynı kare -> ACK -> sonraki GET`

## Mimari

Üç bağımsız thread vardır:

1. `Producer-Network-IN`: Kendi `asyncio` event loop'u ve HTTP istemcisiyle frame metadata/görselini alır, `Input Queue` içine koyar.
2. `Worker-AI-Engine`: Input Queue'dan alır; native inference'ı ayrı bir process'e yollar. Process timeout/crash durumunda sonlandırılır, yeniden kurulur ve aynı kare için finite fallback üretilir.
3. `Consumer-Network-OUT`: Pydantic ile son validasyonu yapar ve sonucu POST eder.

Input ve Output Queue `queue.Queue(maxsize=3)` kullanır. Producer, Consumer'dan doğru frame ACK'i gelmeden yeni GET yapmaz. Böylece thread'ler ayrıdır ancak şartnamedeki “POST olmadan sonraki frame alınamaz” kuralı ihlal edilmez.

GET öncesinden POST ACK sonrasına kadar tüm aşamalar ölçülür. Kare metrikleri JSONL olarak `logs/metrics.jsonl`, uygulama logları dönen `system.log` dosyasına yazılır. `HURGOR_LOG_EVERY` yalnızca periyodik ilerleme logunu sınırlar; metrik kaybı oluşturmaz.

## Kurulum

Python 3.11 veya üzeri gerekir:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,ai]'
cp .env.example .env
pytest
```

Yarışma paketi için hedef Python `.python-version` ile 3.12'ye sabitlenmiştir. Mevcut
ortamı silmeden kontrol ederek kurmak için `scripts/bootstrap_venv.sh`, çevrimdışı
wheelhouse hazırlamak için `scripts/setup_offline.sh` kullanılır.

## Mock sunucu

Sentetik frame'lerle:

```bash
hurgor-mock-server --port 8765
```

Yerel videoyu sırayla frame'lere bölerek:

```bash
hurgor-mock-server --video /absolute/path/test.mp4 --port 8765

# Etiketli test splitindeki gerçek görüntüleri sırayla sunmak için:
hurgor-mock-server --image-dir /absolute/path/images/test --port 8765
```

2026 örnek verisini yarışma hızına daha yakın 7.5 FPS mantığıyla çalıştırmak için
30 FPS videoda `--frame-stride 4` kullanılabilir:

```bash
hurgor-mock-server \
  --video /Users/yusufkaya/Desktop/HürGör/veri/THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_1.MP4 \
  --translation-csv /Users/yusufkaya/Desktop/HürGör/veri/THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_1_translation.csv \
  --frame-stride 4 \
  --healthy-frames 450 \
  --port 8765
```

Durum ve API dokümanı:

- `http://127.0.0.1:8765/api/status`
- `http://127.0.0.1:8765/docs`

Fault injection ayarları:

- `HURGOR_MOCK_CORRUPT_EVERY=10`
- `HURGOR_MOCK_EMPTY_EVERY=15`
- `HURGOR_MOCK_EMPTY_IMAGE_EVERY=...`
- `HURGOR_MOCK_GET_DELAY_MS=...`
- `HURGOR_MOCK_POST_DELAY_MS=...`
- `HURGOR_MOCK_TOKEN_EXPIRE_AFTER_REQUESTS=...`
- `HURGOR_MOCK_RATE_LIMIT_EVERY=...`
- `HURGOR_MOCK_SERVER_ERROR_EVERY=...`
- `HURGOR_MOCK_SERVER_ERROR_STATUS=500`

Mock, yerel taslak endpointlerine ek olarak canlı 2026 arayüzünü taklit eden
`/auth/`, `/progress/`, `/frames/`, `/translation/`, `/reference/`, `/classes/`,
`/media/...` ve `/prediction/` endpointlerini sunar.

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

Resmî bağlantı arayüzü için `.env` dosyasına komitenin verdiği değerleri yazın:

```dotenv
TEAM_NAME=...
PASSWORD=...
EVALUATION_SERVER_URL=http://havaciliktayapayzeka.teknofest.org:1025/
SESSION_NAME=ONLINE_YARISMA_2026
HURGOR_AUTH_SCHEME=auto
HURGOR_AUTH_TOKEN=
HURGOR_TOKEN_ENDPOINT=/auth/
```

Şifreyi repoya eklemeyin. Bağlantıyı POST atmadan kontrol etmek için:

```bash
hurgor-official-probe
```

10 Temmuz 2026 bağlantı kontrolünde resmî Takım Bağlantı Arayüzü incelendi.
Token elle bulunmaz; istemci `TEAM_NAME/PASSWORD` ile otomatik olarak
`POST /auth/` çağırır, dönen token'ı bellekte tutar ve sonraki istekleri
`Authorization: Token ...` başlığıyla yapar. Token loglanmaz ve Git'e yazılmaz.

Prob başarılıysa tek frame canlı test:

```bash
hurgor-client --max-frames 1
```

Resmî mod model-siz başlamaz. Yalnızca bilinçli haberleşme dry-run'ında
`HURGOR_ALLOW_NOOP_DETECTOR=true` verilebilir; bu modun çıktısı yarışma AI sonucu değildir.

Ön kontrol:

```bash
hurgor-preflight
hurgor-preflight --network
# GET frame yapar ama POST yapmaz; sunucuda outstanding frame bırakabilir:
hurgor-preflight --network --fetch-frame
```

Yerel mock sözleşmesine göre:

- GET cevabı tek elemanlı JSON listesidir ve mock sunucu `health_status` üretir.
- İstemci, dokümandaki metin/şekil farkı nedeniyle hem `health_status` hem
  `gps_health_status` okuyabilir.
- POST gövdesi her zaman tek elemanlı JSON listesidir.
- `user`, `HURGOR_USER_URL` değerinden mutlak URL olarak POST'a eklenir.
- `session`, `HURGOR_SESSION_URL` ile mock GET cevabında üretilir ve gelen oturumla
  karşılaştırılır; resmî POST örneğinde `session` alanı bulunmadığı için POST'a eklenmez.
- Nesne sınıfları `http://SERVER/classes/CLASS_ID/` biçiminde gönderilir.
- Tahmin kimliği deterministik SHA-256 özetinden, JSON için güvenli tamsayı aralığında üretilir.

Resmî 2026 Takım Bağlantı Arayüzü sözleşmesine göre:

- Login: `POST /auth/`
- İlerleme: `GET /progress/`
- Frame: `GET /frames/`
- Translation: `GET /translation/`
- Tahmin: `POST /prediction/`
- Referanslar: `GET /reference/`
- Frame görseli: `/media` önekiyle indirilir.
- POST gövdesi tek dict'tir; `id` ve `user` gönderilmez.
- Nesne sınıf URL'leri resmî arayüzde `classes/1..4` şeklindedir.

## Detector veri hazırlığı

Hazır dış kaynak listesi `configs/external_datasets.json` içindedir. API anahtarı
istemeyen parçaları indirmek ve normalize dataset raporu üretmek için:

```bash
PYTHON_BIN=.venv/bin/python scripts/prepare_external_datasets.sh
```

Bu script Roboflow exportlarını ve 46 GB Google Drive arşivini bilinçli olarak
atlar. Roboflow API key geldiğinde:

```bash
export ROBOFLOW_API_KEY="..."
.venv/bin/python tools/download_external_assets.py --only roboflow_hepsi_aaa
.venv/bin/python tools/download_external_assets.py --only roboflow_teknofest_2023_uap_uai
.venv/bin/python tools/build_detector_dataset.py --clean
.venv/bin/python tools/resplit_yolo_dataset.py artifacts/datasets/hurgor_detector artifacts/datasets/hurgor_detector_grouped --clean
```

2022 Eflatun/GitHub kaynağının gerçek `dataset.zip` dosyası Google Drive üzerinde
yaklaşık 46 GB görünmektedir. Mevcut diske sığmadığı için bu arşiv otomatik
indirilmez; harici disk veya en az 55-60 GB boş alan gerekir.

## Model runtime ve export

Runtime `.pt`, `.pth` veya `.h5` yüklemez. YOLO modeli ONNX'e çevrilir:

```bash
python -m pip install -e '.[export]'
hurgor-export-yolo weights/best.pt --target onnx --image-size 640
```

Sonra `.env` içinde:

```dotenv
HURGOR_YOLO_ONNX_PATH=/absolute/path/best.onnx
HURGOR_MODEL_MANIFEST_PATH=/absolute/path/best.json
HURGOR_MODEL_SHA256=<onnx-sha256>
HURGOR_DETECTOR_THRESHOLDS_PATH=/absolute/path/detector_thresholds.json
HURGOR_THERMAL_SPECIALIST_ONNX_PATH=/absolute/path/thermal-best.onnx
HURGOR_THERMAL_SPECIALIST_MANIFEST_PATH=/absolute/path/thermal-best.json
HURGOR_THERMAL_SPECIALIST_SHA256=<thermal-onnx-sha256>
HURGOR_THERMAL_SPECIALIST_CONFIDENCE=0.10
HURGOR_THERMAL_SPECIALIST_TIMEOUT_MS=200
HURGOR_THERMAL_SPECIALIST_SLOW_THRESHOLD_MS=180
HURGOR_THERMAL_SPECIALIST_COOLDOWN_FRAMES=30
HURGOR_THERMAL_SPECIALIST_COOLDOWN_SECONDS=20
HURGOR_REFERENCE_IMAGES_DIR=/absolute/path/references
```

Varsayılan ONNX Runtime provider sırası `TensorRT -> CUDA -> CoreML -> CPU` şeklindedir ve `HURGOR_ONNX_PROVIDERS` ile sabitlenebilir. Model manifesti sınıf sırasını (`arac, insan, uap, uai`), output formatını ve SHA-256 değerini doğrular. “3 kat hız” sabit bir garanti değildir; hedef donanımda benchmark ile doğrulanmalıdır.

İki sınıflı termal uzman yapılandırılırsa RGB karelerde ana model tek başına
çalışır. Termal karelerde `insan` uzman modelden; `arac`, `uap` ve `uai` ana
modelden alınır. Uzman model hata verirse veya sistem degraded moda girerse tam
dört sınıflı ana modele otomatik dönülür. Uzman çağrısı 200 ms'yi aşarsa ana
sonuç bekletilmeden kullanılır ve uzman 30 kare/20 saniye dinlendirilir. CoreML
çalıştırmalarında uzman CPU'ya izole edilir; NVIDIA hedefte TensorRT/CUDA seçimi
otomatik kalır. Bağımsız ölçüm ve sınırlar
[`docs/THERMAL_HUMAN_FUSION_2026-07-15.md`](docs/THERMAL_HUMAN_FUSION_2026-07-15.md)
dosyasındadır.

Sınıfa özel confidence profili de ONNX SHA-256 değerine bağlanır. Başka ağırlıkla
üretilmiş veya sınıf sırası farklı bir profil runtime tarafından reddedilir. Validation
eğrilerinden yeni profil üretmek için:

```bash
PYTHONPATH=src .venv/bin/python tools/calibrate_detector_thresholds.py \
  --weights colab_result/best.pt \
  --runtime-model colab_result/best.onnx \
  --data artifacts/datasets/hurgor_detector_grouped/data.yaml \
  --class-name insan --device mps \
  --output colab_result/detector_thresholds.json
```

AI modülleri:

- YOLO uyumlu ONNX detector; sınıf bazlı confidence, class-aware NMS ve yüksek-IoU
  çapraz sınıf duplicate koruması
- Yoğun topolojik gürültü için çok eşikli, persistence-inspired connected-component ön filtresi
- UAP/UAİ ve engeller için ters izdüşüm tabanlı 3B frustum/AABB IoU analizi; görüntü
  kenarına değen alanı ve insan/araç/referans obje kesişimini emniyetsiz sayan ikinci kontrol
- GPS sağlıklı karelerden kamera ekseni/piksel→metre dönüşümünü öğrenen; çift yönlü
  Lucas-Kanade, RANSAC homografi, dönüş dengelenmesi, opsiyonel normalize quaternion
  yaw telemetrisi ve `SE(3)` birikimi kullanan odometri
- RGB/termal normalize edilmiş ORB özellikleri, RANSAC homografi ve `detected_undefined_objects`
- Canlı `/reference/` kayıtlarını indirip hashleyen, aktif frame aralığı dışında eşlemeyi kapatan dinamik referans yöneticisi
- UAP/UAİ landmarklarını önceleyen RANSAC affine kamera-hareketi ayrıştırması ve
  yetersiz kanıtta yanlış hareket etiketi yerine `-1` güvenli durumu

Önemli sınırlar:

- Hafif topolojik filtre tam persistent-homology/TDA kütüphanesi değildir.
- ORB çapraz spektral eşlemenin yalnızca baseline'ıdır. Üretim matcher'ı LLVIP'nin
  herkese açık RGB/termal örneklerinde iki yönde 0/32 başarı verdiği için Görev 3
  tamamlanmış kabul edilmez. Resmî XoFTR checkpoint'i aynı smoke testinde 32/32
  geometrik başarı verdi; ancak METU-VisTIR resmî demo çiftlerinde 480/640 px'de
  yalnız 1/2 kapıyı geçti ve Mac MPS 480 px p95 süresi bir yönde 868 ms oldu.
  Bu nedenle XoFTR güçlü adaydır fakat henüz üretim matcher'ı değildir. NVIDIA
  hedefte süre/yanlış-eşleşme/IoU kapısı ve uçtan uca regresyon geçmeden
  `UndefinedObjectMatcher` adaptörüyle değiştirilmemelidir.
- Monoküler görsel odometride mutlak ölçek doğrudan gözlenemez; yeni motor metrik ölçeği
  GPS sağlıklı ardışık karelerin gerçek X/Y/Z farklarından çevrimiçi öğrenir.
- Yeni kalibre odometri gerçek hava RGB kaydında dört kesinti penceresini geçti. Tam
  450 sağlıklı + 1800 kesintili AIR oturumunda aktif projektif özellik profiliyle,
  yönelim alanları yokken bütünleşik MAE 33,60 m; opsiyonel quaternion yönelimi
  varken 24,22 m; son-konumu-tut tabanında 205,25 m ölçüldü. Genel doküman örnek
  veride yönelim vaat etse de teknik API tablosu bu alanları garanti etmez; sistem
  yönelim olmadan çalışmaya devam eder. Gerçek termal
  motion-capture stres kaydı yarışma benzeri hava
  hareketi içermediği ve çevrimiçi kapıyı geçemediği için termal doğruluk hâlâ
  kanıtlanmış değildir. `HURGOR_ENABLE_EXPERIMENTAL_VO=true` iken dahi zaman-sıralı
  doğrulama kapısını geçemeyen akış hareket üretmez ve güvenli son-konum davranışına
  döner; bu nedenle doğrulanmış RGB kazancı yarışma yapılandırmasında açılmıştır.
- Aktif detector gerçek HIT-UAV termal testinde araç için mAP50 0,344, insan için
  0,032 verdi. Termal algılama yarışma kalitesinde değildir ve ayrı termal eğitim
  verisi olmadan tamamlanmış sayılmaz.
- 3B analiz iç karardır; resmî JSON yalnızca 2B kutuları gönderir.

Model veya görüntü bozulursa boş tespit ve son güvenilir/0 konumla geçerli fallback JSON gönderilir. JSON'a `NaN` yazılmaz; standart JSON ve Pydantic doğrulaması finite sayı ister.

### Görsel odometri doğruluk kapısı

Deterministik RGB/termal regresyon videosu üretip GPS kesintisini sınamak için:

```bash
PYTHONPATH=src .venv/bin/python tools/generate_odometry_fixture.py \
  --output-dir artifacts/odometry/synthetic-rgb --modality rgb
PYTHONPATH=src .venv/bin/python tools/evaluate_odometry.py \
  artifacts/odometry/synthetic-rgb/odometry-rgb.avi \
  artifacts/odometry/synthetic-rgb/translation-rgb.csv \
  --dropout-start 100 --dropout-end 220 --require-gate
```

Harici veya resmi yarışma verisinde aynı araç gerçek video ve
`translation_x,translation_y,translation_z,frame_numbers` kolonlu CSV ile çalıştırılır.
Varsa `orientation_x,orientation_y,orientation_z,orientation_w` kolonlarının dördü
birlikte opsiyonel olarak kullanılır; kısmi quaternion kabul edilmez.
Araç aday MAE'yi son-konumu-tut baseline'ıyla karşılaştırır; 103 m referansını, 50 m ilk
hedefini, kalibrasyon hazırlığını, P95 `< 800 ms` süreyi ve GPS geri geldiğindeki
yeniden sabitlemeyi tek JSON raporunda doğrular. Güncel gerçek veri sonucu:
[`docs/POSITION_ESTIMATION_REAL_DATA_2026-07-14.md`](docs/POSITION_ESTIMATION_REAL_DATA_2026-07-14.md).
Sentetik fazın tarihsel kaydı:
[`docs/POSITION_ESTIMATION_PHASE2_2026-07-13.md`](docs/POSITION_ESTIMATION_PHASE2_2026-07-13.md).

### Çapraz-spektral matcher aday kapısı

Üretim ORB'sini değiştirmeden önce harici resmî XoFTR checkout/checkpoint'ini
LLVIP ve XoFTR'nin METU-VisTIR demo çiftleri üzerinde ölçmek için:

```bash
python -m pip install -e '.[matching-eval]'
PYTHONPATH=src python tools/evaluate_xoftr_candidate.py \
  --xoftr-repo /path/to/XoFTR \
  --checkpoint /path/to/weights_xoftr_640.ckpt \
  --llvip-figure artifacts/evaluation/reference_matching_sources/llvip-figure1.png \
  --output artifacts/evaluation/reference-matching-xoftr-candidate.json \
  --device auto --resize 480
```

Araç kaynak commit'ini ve checkpoint SHA-256'sını rapora yazar. Çıktı yarışma mAP'i
değildir; yalnız aday eleme kapısıdır. Hedef NVIDIA cihazda yanlış eşleşme, kutu IoU,
YOLO dahil uçtan uca p95 ve 2250-kare regresyon geçmeden üretime alınmamalıdır.

15 Temmuz 2026 izole yeterlilik koşullarında aktif ONNX model, ORB referans aşaması
ve odometri birlikte açıkken projektif profil iki tam oturumda doğrulandı. Yönelim
alanları olmadan 2250/2250 cevap, sıfır fallback/SLA/restart, 239,12 ms uçtan uca
p95 ve 1800-kare kesintide 33,60 m MAE elde edildi. Opsiyonel quaternion yönelimiyle
ayrı tam oturum yine 2250/2250 ve sıfır fallback/SLA/restart verdi; uçtan uca p95
307,41 ms, azami süre 701,94 ms, duvar saati hızı 5,72 FPS ve kesinti MAE'si
24,22 m oldu. İlk 50 m kapısı geçilmiş, 10–20 m final hedefi ve yarışma benzeri
termal doğrulama henüz geçilmemiştir. Makine başka ağır iş yükleriyle paylaşılırsa
süre garantisi geçerli değildir; devre kesici böyle bir durumda protokolü açık
tutar fakat AI çıktısının yerine kontrollü fallback gönderir.

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

## Güvenli yarışma scriptleri

```bash
scripts/preflight.sh
scripts/run_mock.sh
scripts/run_official_probe.sh
# POST başlatmak için bilinçli onay bayrağı zorunludur:
scripts/run_official.sh --confirm-official --max-frames 1
scripts/run_endurance.sh 2250
scripts/collect_metrics.sh logs/metrics.jsonl
```

RGB ve termal localhost smoke/endurance testleri için mock modality ayrı
ayarlanabilir:

```bash
HURGOR_MOCK_MODALITY=rgb HURGOR_MOCK_VIDEO_NAME=hurgor_mock_rgb \
  python -m hurgor.mock_server --host 127.0.0.1 --port 5124
HURGOR_ENDURANCE_URL=http://127.0.0.1:5124 \
  HURGOR_METRICS_FILE=logs/metrics-rgb.jsonl scripts/run_endurance.sh 100

HURGOR_MOCK_MODALITY=thermal HURGOR_MOCK_VIDEO_NAME=Ornek-Veri-2-Termal \
  python -m hurgor.mock_server --host 127.0.0.1 --port 5125
HURGOR_ENDURANCE_URL=http://127.0.0.1:5125 \
  HURGOR_METRICS_FILE=logs/metrics-thermal.jsonl scripts/run_endurance.sh 100
```

`reference_predictions` alt şeması 13 Temmuz 2026'da resmî sunucunun doğrulama cevabıyla
teyit edilmiştir. ORB/RANSAC çıktısı, `/reference/` manifestindeki gerçek `reference` URL'si
ve bounding-box koordinatlarıyla resmî POST şemasına dönüştürülür.
