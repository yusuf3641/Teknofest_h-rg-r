# Termal İnsan Uzmanı Birleşim Kararı — 15 Temmuz 2026

## Uygulanan politika

- RGB kare: dört sınıfın tamamı ana YOLO26n modelinden alınır.
- Termal kare: `arac`, `uap`, `uai` ana modelden; `insan` iki sınıflı termal
  uzman YOLO26n modelinden alınır.
- Uzman model başlatma veya inference hatası: kare atlanmaz; ana modelin bütün
  sınıfları kullanılarak geçerli JSON üretilir ve uzman oturum boyunca kapatılır.
- Graceful-degradation modu: SLA'yı korumak için ikinci model atlanır ve yalnızca
  ana model çalışır.
- Uzman çağrısı 200 ms'de kesilir. Geç kalan çağrı ana sonucu bekletmez; uzman
  30 kare ve en az 20 saniye cooldown'a alınır.

Modeller SHA-256 ve sınıf sırası içeren ayrı manifestlerle açılışta doğrulanır.
Termal uzmanın resmi sınıf eşlemesi kesin olarak `[arac, insan]` olmalıdır.

## Bağımsız doğruluk testi

Test verisi, eğitim paketinden bağımsız 579 termal HIT-UAV karesidir. Bu sette
2.169 araç ve 2.611 insan etiketi vardır. Ana model, uzman model ve sınıf bazlı
birleşim aynı görüntüler, aynı 640 piksel giriş ve aynı değerlendirme koduyla
karşılaştırılmıştır.

| Çalışma | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Ana model | 0,391 | 0,173 | 0,183 | 0,0869 |
| Termal uzman tek başına | 0,392 | 0,192 | 0,151 | 0,0716 |
| Sınıf bazlı birleşim | **0,439** | **0,211** | **0,198** | **0,0882** |

Birleşim ana modele göre precision'ı %12,5, recall'u %21,7, mAP50'yi %8,5 ve
mAP50-95'i %1,6 artırdı. Araç mAP50 değeri ana modeldeki 0,338 düzeyinde
korundu. İnsan mAP50 değeri 0,0274'ten 0,0584'e çıktı; bu aynı ölçüm altında
2,13 kat iyileşmedir.

Tekrarlanabilir ham rapor:
`artifacts/evaluation/thermal-human-fusion-hit-uav-20260715.json`

## Confidence kararı

Bağımsız insan eğrisinde en yüksek F1 yaklaşık 0,038 eşiğinde oluştu. Ancak bu
eşik daha fazla yanlış pozitif ve daha yüksek post-process yükü getiriyor.
Yarışma verisi görülmeden dış veri setine aşırı uyum sağlamamak için çalışma
eşiği **0,10** olarak seçildi:

- 0,10: precision 0,236, recall 0,118, F1 0,158
- 0,25: precision 0,283, recall 0,0876, F1 0,134

0,10 eşiği, 0,25'e göre insan recall'unu yaklaşık %35 ve F1'i yaklaşık %18
artırırken aşırı düşük 0,04 eşiğine göre daha kontrollü çıktı üretir.

## Süre ölçümü

İlk 60 gerçek termal karelik ONNX runtime ölçümünde birleşim:

- P50: 92 ms
- P95: 97 ms
- Maksimum: 138 ms

Sistem yükü altındaki 100 karelik eşik testinde 0,10 için P50 355 ms ve P95
437 ms ölçüldü. Tek bir 938 ms uç değer görüldü; P95 kabul sınırı olan 800 ms'nin
altındadır. Beş ardışık yavaş kare oluşursa mevcut graceful-degradation mekanizması
uzman modeli otomatik atlar.

Ham süre raporları:

- `artifacts/evaluation/thermal-human-fusion-runtime-20260715.json`
- `artifacts/evaluation/thermal-human-fusion-threshold-runtime-20260715.json`

Sağlayıcı benchmark'ında iki modelin sıralı birleşimi için sonuçlar:

| Sağlayıcı | P50 | P95 | Maksimum |
|---|---:|---:|---:|
| CPU, 1 thread | 272 ms | 285 ms | 301 ms |
| CPU, 2 thread | 204 ms | 233 ms | 297 ms |
| CPU, 4 thread | 276 ms | 359 ms | 450 ms |
| CoreML + CPU fallback | **123 ms** | **129 ms** | **148 ms** |

Ana model için sağlayıcı zorlaması kaldırılmıştır. Boş ayar ONNX Runtime'ın
`TensorRT -> CUDA -> CoreML -> CPU` sırasıyla seçim yapmasını sağlar. Uzun Mac
stresinde iki CoreML oturumunun birbirini aç bırakabildiği görüldüğü için runtime,
ana modeli CoreML'de tutarken termal uzmanı tek iş parçacıklı CPU oturumuna otomatik
izole eder. NVIDIA hedefte bu özel CoreML kuralı çalışmaz ve TensorRT/CUDA otomatik
seçimi korunur. Ham rapor:
`artifacts/evaluation/thermal-human-fusion-provider-benchmark-20260715.json`.

## Döngü dayanıklılığı

- 200 kare CPU temiz prova: 200/200 POST kabulü, 0 fallback, 0 SLA ihlali,
  0 deadlock; P95 uçtan uca 259 ms.
- 200 kare CPU kaos provası: 200/200 POST kabulü; 401, 429, 500, bağlantı
  kopması, boş ve bozuk kare enjeksiyonlarının tamamı kontrollü işlendi; yalnız
  enjekte edilen altı karede beklenen fallback kullanıldı, P95 278 ms.
- Maksimum-hız Mac/CoreML deneyi sırasında aynı anda çalışan HürGör dışı bir veri
  işlemi yaklaşık %95 CPU kullandı. Ana model tek başına da 400 karede bir watchdog
  timeout ürettiğinden bu koşu termal uzman regresyonu sayılmadı. Yarışma kabul
  testi temiz hedef cihazda ve arka plan yükleri kapalıyken tekrarlanmalıdır.

Kanıt dosyaları:

- `artifacts/qualification/thermal-human-fusion-smoke-20260715/clean/result.json`
- `artifacts/qualification/thermal-human-fusion-smoke-20260715/chaos/result.json`
- `artifacts/qualification/thermal-main-only-baseline-20260715/clean/result.json`

## Sınırlar ve sonraki kapı

- HIT-UAV setinde UAP/UAİ etiketi yoktur; bu iki sınıf bu testte ölçülmemiştir ve
  ana modelden gelmeye devam eder.
- Termal uzmanın kendi 21 görüntülük test bölümü aynı kaynak videodan geldiği için
  tek başına güvenilir başarı kanıtı sayılmamıştır.
- 0,10 eşiği resmi yarışma örnekleri geldiğinde tekrar kalibre edilmelidir.
- Birleşim, arka plan yükü bulunmayan hedef cihazda 2.250 kare temiz ve kaos
  oturumundan geçirilmeden nihai yarışma modeli olarak dondurulmamalıdır.
