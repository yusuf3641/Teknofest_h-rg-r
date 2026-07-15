# HürGör uçtan uca teknik denetim ve şartname uygunluk raporu

Tarih: 15 Temmuz 2026
Kapsam: Kaynak kod, 2026 genel/teknik şartnameleri, aktif model, gerçek ve sentetik
test verileri, localhost sözleşme sunucusu ve kontrollü resmî sunucu testi.

## 1. Açık karar

Haberleşme omurgası ve RGB-GPS kesintisi senaryosu teslim edilebilir seviyededir;
projenin tamamı maksimum puan hedefi için henüz hazır değildir.

- **Çalışma zamanı/protokol:** GO. En güncel temiz 2250 kare oturumunda 2250/2250
  kabul, sıfır fallback, sıfır SLA ihlali, sıfır restart, deadlock/fatal hata yok;
  uçtan uca p95 307,41 ms, azami süre 701,94 ms ve duvar saati hızı 5,72 FPS'tir.
- **Pozisyon kestirimi:** RGB hava verisi için koşullu GO. Teknik API'de tanımlı
  olmayan yönelim alanları gelmediğinde tam bütünleşik 1800 kare kesinti MAE'si
  33,60 m'dir. Genel dokümanda vaat edilen quaternion yönelimi mevcut olduğunda
  aynı koşulda MAE 24,22 m'ye iner; son-konumu-tut tabanı 205,25 m'dir. İlk 50 m
  kapısı iki durumda da geçilmiş, 10–20 m final hedefi henüz geçilmemiştir.
  Yarışma benzeri termal hava verisinde doğruluk kanıtlanmadığı için tüm spektrumlar
  için tamamlandı denemez.
- **Nesne tespiti:** RGB için kullanılabilir fakat insan sınıfı zayıftır. Termal
  genelleme yarışma seviyesinde değildir.
- **Referans eşleme:** NO-GO. Üretim ORB eşleyici gerçek RGB/termal LLVIP örnek
  çiftlerinde 0/32 başarı vermiştir. Resmî XoFTR adayı aynı smoke testinde 32/32
  üretmiştir; ancak METU-VisTIR'da 1/2, 480 px Mac p95'te en kötü yönde 868 ms ve
  henüz yarışma runtime'ına entegre değildir.
- **Nihai teslim:** Protokol denemesi yapılabilir; bütün puan kalemlerinde iddialı
  final teslimi için termal detector, insan sınıfı, hareket/iniş etiketleri ve
  çapraz-modal referans eşleme tamamlanmalıdır.

Özet aşama görünümü:

| Alan | Karar | Kanıtlanan sınır |
|---|---|---|
| Protokol ve çalışma zamanı | GO | 2250/2250, p95 307,41 ms, fatal/deadlock/fallback 0 |
| Görev 1 RGB detector | Kısmi GO | Genel mAP50 0,828; insan mAP50 0,548 |
| Görev 1 termal detector | NO-GO | HIT-UAV toplam mAP50 0,188; insan 0,032 |
| Görev 2 RGB pozisyon | Koşullu GO | Yönelimsiz 33,60 m; yönelimli 24,22 m MAE |
| Görev 2 termal pozisyon | NO-GO/kanıt eksik | Güvenlik kapısı hatalı drift'i engelliyor; doğruluk kanıtı yok |
| Görev 3 referans altyapısı | GO | Canlı 5 referans indirildi ve pencere yönetimi çalıştı |
| Görev 3 eşleme kalitesi | NO-GO | Üretim ORB 0/32; XoFTR aday, runtime'a entegre değil |
| Hedef edge cihaz dağıtımı | Bekliyor | Mac CPU geçti; NVIDIA hedef cihaz ölçülmedi |
| Yüksek puanlı final teslim | NO-GO | Termal, insan, hareket/iniş ve Görev 3 açıkları sürüyor |

Bu karar, kodun varlığına değil aşağıdaki ölçümlere dayanır.

## 2. Şartnameden doğrulanan ana gereksinimler

Teknik şartnameye göre ağırlıklar: Görev 1 nesne tespiti %25, Görev 2 pozisyon
kestirimi %40, Görev 3 görüntü eşleme %25, final raporu %5 ve sunum %5'tir.

Kritik teknik hükümler:

- Tipik oturum 5 dakika, 7,5 FPS ve 2250 karedir; her kare için ayrı sonuç beklenir.
- Yarışma öncesi test oturumu 2 dakika/900 kare olarak tarif edilir ve puanlanmaz;
  yarışma oturumu 2250 karedir. Çevrimiçi simülasyon bölümü açıkça nesne tespiti ve
  pozisyon kestirimini ister; Görev 3 final yarışma için yine kritik puan kalemidir.
- Yarışma platformu için saniyede en az 1 kare işleyebilen donanım yeterlidir; hız
  doğrudan puan kriteri değildir.
- İlk pozisyon `(0, 0, 0)` olmalıdır. İlk 450 karede konum sağlıklıdır; kalan 1800
  karede sağlık değeri bilinmeyen zamanda sıfıra geçebilir.
- Sağlık sıfırken yalnız görüntüden X/Y/Z konumu üretilmelidir. Puan, kare başına
  üç boyutlu Öklid hatasının ortalamasına bağlıdır.
- Araç, insan, UAP ve UAİ kutuları IoU 0,5 ile puanlanır. Yinelenen kutular,
  hatalı araç hareketi ve hatalı iniş durumu AP'yi düşürür.
- UAP/UAİ kısmen görünse bile tespit edilmeli; tamamı karede değilse veya üzerinde
  insan, araç ya da yabancı nesne varsa `landing_status=0` gönderilmelidir.
- Görüntü RGB veya termal, FHD veya 4K olabilir; donma, bozulma ve tamamen kayıp
  kareler görülebilir.
- Referans nesne farklı kamera/spektrum, açı, irtifa veya uydu görüntüsünden gelebilir.
- Oturum başında verilen referansların tamamı akışta görünmek zorunda değildir.
- Şartname metni `gps_health_status`, Şekil 16 ise `health_status` alanını kullanır ve
  sağlıksız konumu `"NaN"` metniyle örnekler. İstemci iki alan adını da kabul eder,
  NaN girişini sağlık=0 olarak normalize eder ve POST JSON'una hiçbir zaman NaN yazmaz.

## 3. Mevcut mimari ve tamamlanan bileşenler

### Haberleşme

- `Producer-Network-IN`, `Worker-AI-Engine` ve `Consumer-Network-OUT` ayrıdır.
- Girdi/çıktı kuyrukları `queue.Queue(maxsize=3)` ile sınırlıdır.
- Üretici, aynı kare için POST ACK alınmadan sonraki GET'i yapamaz.
- HTTP I/O ayrı `asyncio` döngülerinde, native inference ayrı process'te çalışır.
- 401 token yenileme, 429 `Retry-After`, 5xx, bağlantı kopması ve bozuk/boş kare
  yolları kontrollü retry/fallback ile ele alınır.
- Watchdog timeout/crash halinde process'i emekliye ayırır, son güvenli odometri
  durumunu yeni process'e taşır ve POST zincirini açık tutar.
- Art arda timeoutlarda yeniden başlatma fırtınasını engelleyen devre kesici vardır.

İlgili modüller: `src/hurgor/threaded_pipeline.py`, `src/hurgor/watchdog.py`,
`src/hurgor/client.py`, `src/hurgor/mock_server.py`.

### Resmî sözleşme

- `/auth/`, `/progress/`, `/frames/`, `/translation/`, `/reference/`, `/prediction/`
  sözleşmesi ayrı adaptörle desteklenir.
- Tahmin ID'si güvenli tamsayıdır; sınıflar resmî sunucunun `classes/1..4` URL'lerine
  çevrilir; bütün sayısal kutu ve konum değerleri finite olmak zorundadır.
- Yarışma günündeki özel Ethernet IP'si CLI ile değiştirildiğinde resmî endpointler
  korunur; localhost mock çalışması resmî sözleşmeye yanlışlıkla yönelmez.
- Model manifesti, sınıf sırası, ONNX SHA-256 ve confidence profili açılışta doğrulanır.

İlgili modüller: `src/hurgor/models.py`, `src/hurgor/config.py`,
`src/hurgor/client.py`, `src/hurgor/preflight.py`.

## 4. GPS/pozisyon implementasyonunun kod denetimi

Buradaki “GPS” bir telefon GPS SDK'sı değildir. İstemci, yarışma sunucusunun her
kareyle verdiği `translation_x/y/z` ve `gps_health_status` alanlarını tüketir.
Dolayısıyla mobil konum izni, Android/iOS background permission ve harita SDK'sı
bu projeye uygulanamaz. Bunların yarışma karşılığı kimlik doğrulama, kesintisiz
process çalışması, konum sağlığı ve kaynak tüketimidir.

### Çalışma akışı

1. Sağlıklı karede sunucunun X/Y/Z değeri doğrudan güvenli anchor olarak kullanılır.
2. İlk sağlıklı ardışık karelerde çift yönlü Lucas-Kanade özellik akışı çıkarılır.
3. RANSAC homografi ile global kamera dönüşü/projektif etki ayrılır.
4. Görüntü hareketi ile gerçek X/Y/Z farkları arasında robust metrik dönüşüm öğrenilir.
5. Kalibrasyonun son kısmı zaman-sıralı hold-out olarak saklanır. Aday hareket,
   adım ve tüm yörünge düzeyinde son-konumu-tut tabanını geçmeden “qualified” olmaz.
6. Sağlık sıfır olduğunda yalnız qualified model delta üretir; delta SE(3) üzerinde
   biriktirilir, uç değerler ve zayıf homografiler reddedilir.
7. Görsel özellik kaybolursa son güvenli hız kontrollü olarak sönümlenir; rastgele
   sıçrama üretilmez.
8. GPS geri geldiğinde pozisyon kesin sunucu değerine re-anchor edilir.
9. Worker timeout/crash halinde kalibrasyon, pose, önceki görüntü ve hareket durumu
   checkpoint ile yeni process'e taşınır. Parent süreç de kontrollü acil konum
   extrapolasyonu yapabilir.
10. `orientation_x/y/z/w` alanlarının dördü birden ve finite olarak gelirse quaternion
    normalize edilir, ardışık yaw farkı dünya eksenine dönüşümde kullanılır. Alanlar
    yoksa sistem görüntüden kestirilen yaw ile çalışmaya devam eder; kısmi veya bozuk
    quaternion kabul edilmez. Telemetri kesilip geri gelirse aradaki dönüş ikinci kez
    uygulanmaz.

Ana implementasyon: `src/hurgor/odometry.py`. Runtime bağlama:
`src/hurgor/vision.py`, `src/hurgor/inference.py`,
`src/hurgor/threaded_pipeline.py`, `src/hurgor/watchdog.py`.

### GPS testleri ve ölçülebilir sonuçlar

| Test | Veri/koşul | Sonuç | Karar |
|---|---|---:|---|
| İlk konum | İlk kare, sağlıklı GPS | ilk hata 0,000000 m | Geçti |
| Tam GPS kesintisi, yönelim yok | AIR UC200, kare 450–2249, projektif özellik, bütünleşik runtime | MAE 33,60 m; RMSE 37,28 m; p95 52,87 m; final drift 56,15 m | 50 m MAE kapısı geçti |
| Tam GPS kesintisi, opsiyonel quaternion | Aynı AIR oturumu, bütünleşik runtime | MAE 24,22 m; RMSE 26,09 m; p95 32,78 m; final drift 27,42 m | Geçti; 10–20 m hedefi kaldı |
| Doğrudan evaluator, opsiyonel quaternion | AIR UC200, 1800 kare | MAE 25,81 m; RMSE 27,94 m; p95 34,86 m; final drift 30,13 m | Geçti |
| Son-konumu-tut karşılaştırması | Aynı 1800 kare | hold MAE 205,25 m; bütünleşik yönelimli iyileşme %88,20 | Geçti |
| Erken/orta/geç kesintiler | AIR UC200 | MAE 12,61 / 11,27 / 21,23 m | Geçti |
| Farklı gerçek hava verisi | Zurich MAV, projektif özellik | birincil 0,173 m vs hold 0,571 m; erken 0,096 vs 0,207; geç 0,171 vs 0,354 | 3 pencere geçti |
| Zor Zurich orta pencere | Zurich MAV, projektif özellik | 0,262 m vs hold 0,432 m; %39,28 iyileşme | Geçti |
| Termal motion-capture | IRS RTVI | üç pencerede kalibrasyon kapısı hold'a döndü | Hatalı drift yok; hava-termal doğruluğu kanıtlanmadı |
| 2250 kare gerçek döngü, yönelim yok | AIR + aktif YOLO + ORB | 2250/2250; fallback/SLA/restart 0; p95 239,12 ms; outage MAE 33,60 m | Geçti |
| 2250 kare gerçek döngü, yönelimli | AIR + aktif YOLO + ORB + quaternion | 2250/2250; fallback/SLA/restart 0; p95 307,41 ms; outage MAE 24,22 m | Geçti |
| Yönelimli bütünleşik smoke | AIR + aktif YOLO + ORB, 600 kare | 600/600; fallback/SLA/restart 0; p95 108,32 ms; 150-kare outage MAE 2,58 m vs hold 22,21 m | Geçti |
| GPS geri dönüşü | AIR recovery kareleri | re-anchor azami hata 0,0 m | Geçti |
| VO çalışma süresi | AIR yönelimli doğrudan evaluator | p95 11,26 ms | Geçti |

En güncel yönelimli tam oturumun duvar saati hızı 5,72 FPS, uçtan uca p95'i
307,41 ms ve azami süresi 701,94 ms'dir; şartnamedeki en az 1 FPS ve iç kalite
kapısındaki 800 ms sınırları geçilmiştir. Aynı projektif profil yönelim alanları
olmadan da tam 2250 karede doğrulanmıştır. Bu değerler Mac CPU kanıtıdır; hedef
donanımda yeniden ölçülmelidir.

### GPS için açık sınırlar

- Yönelim yokken 33,60 m, opsiyonel yönelim varken 24,22 m; ikisi de ilk 50 m
  hedefini geçer fakat 10–20 m final hedefini geçmez.
- Genel doküman örnek veride pozisyon ve yönelim paylaşılacağını söyler; teknik canlı
  API tablosu yönelim alanlarını tanımlamaz. İyileştirme bu nedenle opsiyoneldir ve
  sistem bu alanlara bağımlı değildir.
- Monoküler mutlak ölçek ilk sağlıklı GPS bölümünden öğrenildiği için kamera/video
  dağılımı değişiminde yeniden kalibrasyon kalitesi belirleyicidir.
- Şartname sağlık bilgisini ikili verir; “düşük sinyal” için ayrı seviye yoktur.
  Sistem sağlık=0 ve bozuk görüntü kombinasyonunu güvenli fallback ile ele alır.
- Yarışma benzeri alt-görüş termal video + gerçek X/Y/Z verisi bulunmadığı için
  termal GPS kesintisi doğruluğu kanıtlanmış değildir.
- AIR üzerinde tam 3B quaternion dönüşümü yaw-only yöntemden yalnız 0,004 m farklı
  sonuç verdi; ters eksen yorumu hatayı 300 m'nin üzerine çıkardı. Yalnız AIR'de
  iyileşen yüksek regularizasyon kronolojik doğrulamada kötüleşti. Bu alternatifler
  genellenebilir kazanç göstermediği için üretime alınmadı.
- Pil/güç tüketimi bu Mac üzerinde ölçülmedi. Mobil izin/background kontrolleri
  uygulanamaz; hedef edge cihazın watt, sıcaklık ve throttling testi gereklidir.

## 5. Detector ve referans eşleme ölçümleri

### Aktif model

Aktif model dört sınıflı, 640 px girişli custom **YOLO26n ONNX** modelidir. Yeni
leakage-safe eğitim modeli, aynı 294 görüntülük görülmemiş holdout üzerinde daha
düşük sonuç verdiği için otomatik olarak aktif edilmemiştir.

| Model | mAP50 | mAP50-95 | İnsan mAP50 | İnsan mAP50-95 |
|---|---:|---:|---:|---:|
| Aktif grouped v1 | 0,828 | 0,667 | 0,548 | 0,215 |
| Yeni leakage-safe v2 | 0,799 | 0,654 | 0,463 | 0,170 |

Bu holdout yarışma gizli test seti değildir; yalnız model seçimi kanıtıdır.

### Termal genelleme

HIT-UAV'nin etiketli 579 termal test görüntüsünde aktif model:

| Sınıf | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Araç | 0,616 | 0,306 | 0,344 | 0,165 |
| İnsan | 0,151 | 0,053 | 0,032 | 0,013 |
| Toplam ölçülebilen | 0,384 | 0,180 | 0,188 | 0,089 |

HIT-UAV'de UAP/UAİ olmadığı için bu sınıfların termal başarısı ölçülemedi. Sonuç,
aktif RGB-ağırlıklı modelin özellikle termal insan tespitinde yeterli olmadığını
kanıtlar.

### Çapraz spektral referans eşleme

Üretimdeki CLAHE + ORB + ratio test + RANSAC homografi, LLVIP'nin herkese açık
Figure 1 çiftlerinde iki yönde çalıştırıldı:

- Termal referans → RGB görüntü: 0/16
- RGB referans → termal görüntü: 0/16
- Toplam: 0/32

Bu küçük test yarışma mAP'i değildir; fakat aynı sahnenin hizalı RGB/termal
çiftlerinde dahi sıfır başarı, ORB'nin Görev 3 için yeterli olmadığını açıkça
gösterir.

Resmî XoFTR deposunun `e0fbea4` revizyonu ve resmî 640 checkpoint'i
(`sha256=6e7f24e5...6af3a67`) ayrıca test edildi:

| Giriş | LLVIP geometrik kapı | METU-VisTIR resmî demo | Mac MPS p95 |
|---|---:|---:|---:|
| 320 px | 32/32 | 0/2 | 548 / 305 ms |
| 480 px | 32/32 | 1/2 | 868 / 689 ms |
| 640 px | 32/32 | 1/2 | 1044 / 2101 ms |

LLVIP kapısı en az 8 homografi iç noktası ve en az %10 iç-nokta oranıdır. METU
örnekleri daha büyük görüş farkına sahiptir; ikinci çiftte bu oran geçilememiştir.
Sonuç XoFTR'nin ORB'den belirgin biçimde daha iyi bir aday olduğunu, fakat hem
genelleme hem süre kapısının henüz birlikte geçilmediğini gösterir. XoFTR ayrı
PyTorch/GPU bağımlılığı ve yaklaşık 45 MB checkpoint gerektirdiğinden üretim ONNX
runtime'ına alınmamış, tekrar üretilebilir aday benchmark aracı eklenmiştir.

## 6. Uygunluk matrisi

| Şartname maddesi | Mevcut durum | İlgili kod/modül | Yapılan test | Test sonucu | Tespit edilen eksiklik | Uygulanan düzeltme | Son doğrulama | Puan etkisi / risk |
|---|---|---|---|---|---|---|---|---|
| 2250 kareye 2250 ayrı sonuç | Tamam | `threaded_pipeline.py`, `mock_server.py` | En güncel yönelimli temiz tam oturum | 2250/2250; duplicate/order/fatal/deadlock 0 | Önceki yük altında timeout | CPU thread kalibrasyonu, watchdog/devre kesici | Temiz v9 geçti | Kritik çalışma riski düşük |
| POST ACK olmadan yeni GET yok | Tamam | `threaded_pipeline.py` | Mock sıra sayaçları | 0 tekrar GET, 0 sıra ihlali | Yok | ACK gate | Geçti | Eleme riski düşük |
| En az 1 FPS, p95 <800 ms iç kalite kapısı | Tamam/Mac | `metrics.py` | 2250 aktif işlem | 5,72 FPS; p95 307,41 ms; max 701,94 ms | Hedef edge cihaz bilinmiyor | CPU-2 deterministik profil | Mac'te geçti | Donanım riski orta |
| Bounded queue/deadlock yok | Tamam | `threaded_pipeline.py` | Tam oturum | maxsize sınırı, threadler kapandı | Yok | Queue/stop sentinel/join | Geçti | Düşük |
| Bozuk/boş/donmuş kare toleransı | Tamam/Protokol | `mock_server.py`, `inference.py`, `watchdog.py` | 2250 kaos + birim testler | 19 bozuk, 10 boş görüntü ve 14 boş metadata; 2250/2250 POST, fatal/deadlock yok | Ağır ortak makine yükünde 96 fallback ve 8 timeout | Yapılandırılmış hata türü, finite fallback, devre kesici, donmuş kare motion=-1 | Protokol geçti; AI-kalite kapısı kaldı | Eleme riski düşük, puan riski orta |
| 401/429/500/bağlantı kopması | Tamam/Protokol | `client.py`, `qualify_runtime.py` | 2250 deterministik kaos | 32×401, 54×429, 39×5xx, 36 kopma; 2250/2250 kabul | Stres koşusu izole değildi; 9 SLA aşımı | Backoff, token refresh, Retry-After, exact fault accounting | Sıra/duplicate/fatal yok | Eleme riski düşük, hedef cihaz süre riski orta |
| Resmî auth ve sözleşme | Tamam | `client.py`, `models.py` | Kontrollü canlı POST + son salt-okunur preflight | HTTP 200; progress 0→1; auth/progress/5 reference/5 class kontrolleri geçti | Sunucu veya yarışma IP'si değişebilir | Official adapter/contract tests | 15 Temmuz canlı ve salt-okunur kontroller geçti | Eleme riski düşük |
| Dinamik yarışma IP/port | Tamam | `client.py` | Unit test | Özel IP'de resmî endpoint korundu | Yok | `ipaddress` tabanlı override | Geçti | Düşük |
| JSON tip/finite doğrulama | Tamam | `models.py` | Pydantic contract testleri | NaN/±Inf reddedildi | Yok | `allow_inf_nan=False` | Geçti | Eleme riski düşük |
| ONNX/manifest/hash | Tamam | `vision.py`, `preflight.py` | Preflight + SHA doğrulama | Aktif model yüklendi | TensorRT yok | ONNX Runtime CPU profili | Geçti | Hız riski düşük/orta |
| 4 sınıf RGB nesne tespiti | Kısmi | `vision.py`, aktif ONNX | Görülmemiş 294 holdout | mAP50 0,828 | İnsan recall/mAP zayıf | Sınıfa özel eşik, duplicate NMS | Model seçimi geçti | Görev 1 %25, risk yüksek |
| RGB veya termal giriş | Eksik kalite | Aynı detector | HIT-UAV termal test | toplam mAP50 0,188 | Termal eğitim/genelleme yetersiz | Termal dataset hazırlama/evaluator | Eksik | Görev 1, kritik yüksek |
| Araç hareketli/hareketsiz | Kısmi | `MotionCompensatedMotionClassifier` | Sentetik kamera-hareket testleri | Mantık geçiyor | Gerçek etiketli hareket benchmarkı yok | Global kamera hareketi ayrıştırma, yetersiz kanıtta -1 | Sadece sentetik | AP riski yüksek |
| UAP/UAİ kısmi görünüm → landing 0 | Kısmi | `OptimizedObjectDetector` | Boundary unit test | Geçti | Gerçek yarışma etiketli ölçüm yok | Kenar teması kontrolü | Unit test geçti | AP riski orta |
| UAP/UAİ üzerinde insan/araç/yabancı nesne | Kısmi | `FrustumProjector`, landing analizi | Sentetik overlap testleri | İnsan/araç/ref obje geçiyor | Genel yabancı nesne detector'ı yok; yalnız eşlenen referans engel olur | 2B+3B overlap kontrolü | Unit test geçti | AP riski yüksek |
| Aynı nesne için duplicate yok | Tamam/Kod | `OptimizedObjectDetector` | Unit test | Same-class NMS geçti | Gizli veride eşik kalibrasyonu bilinmiyor | class-aware NMS | Geçti | Orta |
| İlk konum (0,0,0) | Tamam | `odometry.py` | AIR + runtime mock | ilk hata 0 m | Yok | İlk-frame anchor | Geçti | Görev 2 %40, düşük |
| İlk 450 kareden yön/ölçek öğrenme | Tamam/RGB | `odometry.py` | Kronolojik validation | calibration/navigation ready | Termal hava verisi yok | Hold-out beceri kapısı | RGB geçti | Görev 2, orta |
| GPS=0 iken görüntüden X/Y/Z | Kısmi güçlü | `odometry.py` | AIR 1800 doğrudan + iki ayrı 2250 bütünleşik koşu | Yönelimsiz MAE 33,60 m; opsiyonel quaternion ile 24,22 m; hold 205,25 m; p95 307,41 ms | 10–20 m hedefi, termal ve resmî kamera profili eksik; canlı API'de yönelim garanti değil | Projektif homografi, robust metrik harita, opsiyonel normalize quaternion yaw, SE(3), re-anchor | 2250/2250 iki profilde geçti | Görev 2, yüksek fırsat/orta risk |
| GPS geri gelince yeniden sabitleme | Tamam | `odometry.py` | Recovery kareleri | max hata 0 m | Yok | Kesin anchor reset | Geçti | Görev 2 riski düşük |
| Worker crash sırasında VO devamlılığı | Tamam/Kod | `watchdog.py`, `threaded_pipeline.py` | Timeout/restart unit testleri | State restore ve parent extrapolation geçti | Gerçek cihaz kill testi ayrıca gerekli | Compact checkpoint + ready-before-fetch | Unit test geçti | Orta |
| Oturum referanslarını alma/aktif pencere | Tamam | `references.py` | Mock + resmî canlı bootstrap | Resmî sunucudan 5 referans alındı | İçerik başarısı ayrı sorun | Hash/cache/frame-window | Canlı geçti | Görev 3 altyapısı düşük risk |
| RGB↔termal/farklı açı referans eşleme | Eksik/aday bulundu | `ORBReferenceMatcher`, `evaluate_xoftr_candidate.py` | LLVIP + METU-VisTIR | ORB 0/32; XoFTR LLVIP 32/32, METU 1/2; 480 px en kötü p95 868 ms | Doğruluk+süre aynı profilde geçmiyor; runtime adaptörü yok | Resmî checkpoint ile tekrar üretilebilir benchmark ve 320/480/640 taraması | Aday kanıtlandı, üretim NO-GO | Görev 3 %25, kritik |
| Graceful degradation | Tamam/altyapı | `threaded_pipeline.py` | Forced timeout testleri | Devre kesici zinciri açık tutuyor | Fallback kare puan getirmez | 2 timeout/30 kare cooldown | Geçti | Eleme riskini azaltır, puan riski sürer |
| Log/ölçüm/print kullanmama | Tamam | `logging_utils.py`, `metrics.py` | Log/JSONL inceleme | Aşama p50/p95/p99 ve modality ayrımı var | Uzun dönem disk rotasyonu hedefte izlenmeli | Rotating log + JSONL | Geçti | Operasyon riski düşük |
| Kapalı ağ/çevrimdışı çalışma | Kısmi | wheelhouse scripts, Docker | Statik preflight | Runtime internet API'si kullanmıyor | Hedef OS wheelhouse ve NVIDIA image denenmedi | Offline kurulum scriptleri | Mac CPU geçti | Deployment riski orta |
| Final raporu iddialarının kanıtı | Eksik/Revizyon gerekli | Mevcut rapor PDF | Rapor-kod karşılaştırması | mAP 0,94/0,89/0,96; pozisyon 1,2 m; drift %1,5; hareket %96; gerçek TDA/3B/“milimetrik” eşleme iddiaları mevcut kanıtlarla doğrulanmıyor | Ölçümler ve yöntem adları güncel değil | Bu denetim raporu ve tekrar üretilebilir kanıt yolları üretildi | Final rapor PDF henüz düzeltilmedi | %5, yüksek itibar riski |
| Yarışma sunumu | İncelenmedi | — | Kanıt yok | Test edilemedi | Sunum dosyası yok/tespit edilmedi | — | Eksik | %5, yüksek |

## 7. Bu denetimde uygulanan düzeltmeler

1. Timeout sonrası process yeniden başlatma fırtınasını sınırlayan frame-tabanlı
   devre kesici eklendi.
2. Worker kapalıyken son güvenli hızdan sönümlü parent-process konum extrapolasyonu
   eklendi.
3. Odometri checkpoint/restore akışı ve worker hazır olmadan yeni frame almama
   davranışı doğrulandı.
4. Yarışma günü özel IP değişiminde resmî endpointlerin local mock endpointlerine
   dönmesi engellendi.
5. Çıkış bounding-box alanlarında NaN ve ±Inf Pydantic seviyesinde yasaklandı.
6. Docker Compose model/referans mount yolları gerçek proje dizinlerine düzeltildi.
7. Aktif model CPU thread sayısı yük altında 1/2/4 profilleriyle ölçüldü; en
   deterministik yarışma ayarı 2 thread olarak seçildi.
8. Görsel odometri yalnız zaman-sıralı validation tabanını geçtiğinde hareket
   üretecek biçimde açık yarışma ayarına alındı.
9. Gerçek termal detector benchmarkı ve tekrar üretilebilir LLVIP referans eşleme
   benchmark aracı eklendi.
10. Yeterlilik aracının initialization hata yolundaki tanımsız `pipeline` riski
    giderildi.
11. Resmî sunucuda salt-okunur auth/progress ve kontrollü tek kare POST tekrar
    doğrulandı.
12. Bozuk kare, watchdog timeout'u ve devre-kesici bypass'ı ayrı hata türleriyle
    sayıldı; kaos yeterlilik kapısı yalnız tamamen açıklanabilen fallback'leri kabul
    edecek biçimde sıkılaştırıldı.
13. Odometriye projektif homografi ızgara özellikleri yapılandırılabilir üretim
    profili olarak eklendi; AIR MAE 35,19 m'den 33,75 m'ye indi ve daha önce zor
    Zurich orta penceresi hold'a karşı %39,28 kazanımla geçti.
14. Resmî XoFTR kodu/checkpoint'i değiştirilmeden 320/480/640 px'de LLVIP ve
    METU-VisTIR üzerinde ölçen, checkpoint hash'i ve kaynak revizyonunu kaydeden
    aday benchmark aracı eklendi.
15. Tüm sayıların finite olması, worker hata tipinin process sınırını geçmesi ve
    yeterlilik kabul hesabı için regresyon testleri eklendi.
16. Projektif profil aktifken 600-kare temiz bütünleşik oturum çalıştırıldı; çalışma
    yapılandırması artefakta gömüldü ve 600/600, sıfır fallback, 105,15 ms p95 ile geçti.
17. Genel dokümanda belirtilen fakat teknik API tablosunda bulunmayan
    `orientation_x/y/z/w` alanları opsiyonel ve geriye uyumlu biçimde eklendi;
    normalize/finite doğrulama, heading çıkarımı, state checkpoint/restore ve CSV/mock
    taşıma yolları test edildi.
18. Projektif profil yönelim olmadan 2250 karede 33,60 m MAE; yönelim varken ayrı
    2250 karede 24,22 m MAE ile uçtan uca doğrulandı. İki koşu da 2250/2250,
    sıfır fallback/SLA/restart ile kapandı.
19. Tam quaternion, ters eksen, farklı regularizasyon ve bias/intercept alternatifleri
    karşılaştırıldı. Yalnız tek veri üzerinde kazanç sağlayıp kronolojik veya farklı
    veri doğrulamasını bozan seçenekler üretime alınmadı.

### Mevcut takım raporundaki kanıt uyumsuzlukları

Takım raporu PDF'si ile güncel kod/artefaktlar sayfa bazında karşılaştırıldı:

- PDF dosya sayfası 9'daki Knowledge Distillation ve gerçek 3B IoU anlatımı için
  eğitim/ablation veya resmî 3B çıktı kanıtı yoktur. Kodda yalnız iç güvenlik kararı
  için frustum/AABB yaklaşımı vardır; resmî JSON 2B kutu gönderir.
- Dosya sayfaları 10–12'deki SuperGlue, Wasserstein metriği, TDA/Persistent Homology
  ve milimetrik eşleme ifadeleri mevcut üretim yolunu tarif etmez. Üretim matcher'ı
  ORB'dir; topoloji filtresi persistence-inspired bağlı bileşen filtresidir.
- Dosya sayfası 13'teki araç `0,94`, insan `0,89`, UAP/UAİ `0,96` mAP; hareket
  `%96`; konum `1,2 m` ve drift `%1,5` ifadeleri güncel adil holdout ve AIR
  ölçümleriyle doğrulanmamıştır.
- Dosya sayfası 14'teki olumsuz hava kararlılığı `>%85` ve `1 FPS %100` ifadeleri
  aynı koşulu tekrar üreten ham test artefaktına bağlanmamıştır.

Final raporda bu ifadeler ya ölçüm protokolü ve artefaktla kanıtlanmalı ya da bu
rapordaki doğrulanmış değerlerle değiştirilmelidir.

## 8. Kök nedenler

- Önceki uzun testlerdeki timeoutlar model hatasından değil, aynı Mac'te HürGör
  dışında çalışan sanallaştırma, indeksleme ve başka analiz süreçlerinin CPU
  çekişmesinden kaynaklanıyordu. İzole temiz v6 koşusu 2250 karede sıfır fallback
  ile geçti. Daha sonra bilinçli ağ/görüntü kaosu ağır dış yükle çakıştığında protokol
  yine 2250/2250 kapandı; fakat 8 watchdog timeout'u, 60 devre-kesici bypass'ı,
  96 fallback ve 9 SLA aşımı oluştu. Bu ikinci koşu dayanıklılık kanıtıdır, temiz
  performans yeterliliği değildir.
- Yeni leakage-safe model daha uzun eğitim görmesine rağmen insan ve genel holdout
  metriği geriledi; “yeni” olması “daha iyi” olduğu anlamına gelmedi.
- Termal başarısızlığın ana nedeni RGB ağırlıklı eğitim dağılımı ve termal UAP/UAİ
  örneği eksikliğidir.
- ORB yalnız yerel yoğunluk/kenar benzerliğine dayanır; termal-RGB görünüş farkını
  öğrenmediği için çapraz-spektral eşleme başarısızdır. XoFTR bu farkı öğrenmiş ve
  LLVIP kapısını geçmiş olsa da büyük görüş farkında 2/2 üretmemiş, 640 px profili
  SLA'yı aşmıştır; doğrudan üretime almak yeni bir süre riski yaratır.
- Monoküler VO'da mutlak ölçek gözlenemez; mevcut çözüm ilk sağlıklı GPS bölümünden
  ölçek öğrenerek bu belirsizliği azaltır, fakat yeni kamera/irtifa dağılımına
  genelleme sınırlıdır.
- AIR'in 1800 karelik kesintisinde yalnız görüntüden biriken yaw ile yer-gerçeği
  quaternion yaw'ı arasında yaklaşık 17 derecelik son fark gözlendi. Opsiyonel
  yönelim telemetrisi bu kaynağı azaltarak bütünleşik MAE'yi 33,60 m'den 24,22 m'ye
  düşürdü. Teknik API bu alanı garanti etmediğinden görüntü-yaw yolu korunmaktadır.

## 9. Kalan işler ve nedenleri

Öncelik sırası puan etkisine göre:

1. **Görev 3 (%25):** XoFTR'nin bulunan 480 px hız/doğruluk adayını izole
   Colab/NVIDIA hedefinde gerçek referans kutu IoU'su ile ölç; METU-VisTIR kapısını
   2/2'ye, hedef p95'i YOLO dahil <800 ms'ye getirmeden runtime adaptörüne alma.
   Ardından 2250 kare regresyonu çalıştır. Gerekirse MINIMA'yı aynı kapıyla kıyasla.
2. **Termal Görev 1:** Etiketlenen 2026 termal görüntüleri HIT-UAV ile birleştir;
   video-bazlı leakage-safe split, class balance ve hard-negative içeren detector
   eğit. UAP/UAİ termal örnekleri özel olarak üretilmeli/toplanmalıdır.
3. **İnsan sınıfı:** Küçük/dik-açı insan örneklerini büyüt; yanlış negatif ve
   sahne-bazlı holdout üzerinden confidence eşiğini yeniden kalibre et.
4. **Hareket ve iniş:** Gerçek ardışık video üzerinde araç motion_status ve
   UAP/UAİ occupancy etiketli benchmark üret. Genel yabancı engel sınıfı veya
   segmentasyon/anomali algılayıcı ekle.
5. **Pozisyon:** Yarışma benzeri RGB ve özellikle termal alt-görüş video + gerçek
   X/Y/Z ve resmî kamera parametreleriyle kamera-profili bazlı kalibrasyon yap.
   2250-kare yönelimli ve yönelimsiz tekrarlar tamamlandı; kalan hedef MAE'yi
   yönelim yokken de <20 m yapmak ve termal doğruluğu kanıtlamaktır.
6. **Deployment:** Nihai NVIDIA edge cihazda CUDA/TensorRT/ONNX provider,
   4K/termal decode, sıcaklık, güç ve 2250 kare soak testi yapılmalıdır.
7. **Rapor/sunum:** Final rapordaki kanıtsız 1,2 m, mAP 0,94, gerçek TDA ve 3B IoU
   ifadeleri ya kanıtlanmalı ya da bu rapordaki ölçümlerle düzeltilmelidir.

## 10. Gerçek hedef cihaz test planı

1. Yarışmada kullanılacak cihazı temiz reboot et; başka container/eğitim süreci
   çalıştırma.
2. Ethernet dışında Wi-Fi'ı kapat; komite IP'sini CLI ile ver; `preflight --network`
   ve progress kontrolü yap.
3. GPU/provider listesini, model SHA-256'yı, kamera profilini ve boş disk alanını logla.
4. 4K RGB, 640×512 termal ve bozuk/donmuş karelerden ayrı 2250-kare oturumlar çalıştır.
5. Her oturumda 2250 GET/POST, duplicate/order violation, fallback, restart,
   p50/p95/p99, RSS, GPU RAM, sıcaklık, watt ve throttling ölç.
6. GPS kesintisini erken/orta/geç ve aralıklı pencerelerde uygula; gizli truth ile
   MAE/RMSE/p95/final drift ve re-anchor hatası hesapla.
7. Inference process'ini kontrollü öldür; outstanding kareye POST geldikten sonra
   sistemin state restore ile devam ettiğini doğrula.
8. Etherneti kısa süre çıkar/tak; 401/429/500 ve sunucu restart testlerini uygula.
9. Gerçek termal/RGB referans çiftlerinde kutu IoU/mAP ve yanlış pozitif oranını ölç.
10. Yalnız bütün kalite kapıları geçerse final image/wheelhouse'u hashleyip dondur.

## 11. Kanıt dosyaları

- Önceki temiz tam oturum: `artifacts/qualification/air-vo-v6-final/clean/result.json`
- Projektif yönelimsiz tam oturum: `artifacts/qualification/air-vo-v8-projective-full/clean/result.json`
- Projektif yönelimli 600-kare smoke: `artifacts/qualification/air-vo-v9-orientation-600/clean/result.json`
- Projektif yönelimli tam oturum: `artifacts/qualification/air-vo-v9-orientation-full/clean/result.json`
- Ağır dış yükte kaos oturumu: `artifacts/qualification/air-vo-v6-final/chaos/result.json`
- GPS projektif gerçek hava tam kesinti: `artifacts/odometry/air-uc200-synthetic/evaluation-full-projective-v3.json`
- GPS projektif + quaternion tam kesinti: `artifacts/odometry/air-uc200-synthetic/evaluation-full-projective-orientation-v4.json`
- GPS önceki profil karşılaştırması: `artifacts/odometry/air-uc200-synthetic/evaluation-full-physical-v2.json`
- GPS Zurich pencereleri: `artifacts/odometry/zurich-real/evaluation-*.json`
- Termal GPS güvenlik testleri: `artifacts/odometry/irs-thermal-real/evaluation-*.json`
- Aktif model adil holdout: `artifacts/evaluation/old_model_unseen294_20260713.json`
- Yeni model adil holdout: `artifacts/evaluation/new_model_unseen294_20260713.json`
- HIT-UAV termal detector: `artifacts/evaluation/active-yolo26n-hit-uav-thermal-test-20260715.json`
- LLVIP referans eşleme: `artifacts/evaluation/reference-matching-llvip-orb-20260715.json`
- XoFTR 320 px adayı: `artifacts/evaluation/reference-matching-xoftr-candidate-320-20260715.json`
- XoFTR 480 px adayı: `artifacts/evaluation/reference-matching-xoftr-candidate-480-20260715.json`
- XoFTR 640 px adayı: `artifacts/evaluation/reference-matching-xoftr-candidate-20260715.json`
- CPU/CoreML benchmarkları: `artifacts/evaluation/benchmark-active-*-20260715.json`
- Resmî kontrollü canlı POST: `system.log` ve `logs/metrics.jsonl` içindeki 15 Temmuz
  2026 04:12–04:13 kayıtları; resmî progress 0'dan 1'e ilerlemiştir.
- Son resmî salt-okunur preflight:
  `artifacts/audit/official-preflight-readonly-20260715.json`; config, disk, model,
  provider, auth, progress, 5 reference ve 5 class kontrolü geçmiştir.

## 12. Nihai teslim kararı

**Protokol ve RGB pozisyon test oturumu için hazır. Tüm yarışma puan başlıklarında
yüksek performanslı final teslim için hazır değil.**

Görev 2'nin %40'lık bölümünde anlamlı ve ölçülmüş kazanım vardır. Görev 3 için XoFTR
adayı ORB'nin 0/32 sonucunu LLVIP'de 32/32'ye çıkarmıştır; ancak üretim matcher'ı hâlâ
ORB'dir, METU sonucu 1/2'dir ve süre kapısı sınırdadır. Görev 1'deki termal/insan
zayıflığıyla birlikte bunlar “tam hazır” kararı verilmesini teknik olarak engeller.
Bu alanlar düzeltilmeden yalnız çalışan haberleşme iskeleti, yüksek toplam yarışma
puanı için yeterli olmayacaktır.
