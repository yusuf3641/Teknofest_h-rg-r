# HürGör Nesne Tespiti Faz 3 — 13 Temmuz 2026

## Karar

Yarışma runtime modeli değiştirilmedi. Aktif model hâlâ:

- `colab_result/best.onnx`
- YOLO26n, 640 piksel, `arac / insan / uap / uai`
- SHA-256: `484a766c10fe67c6177dc8ab3379c21ea4d9558a1eb95060b814402f05c1d31f`

Yeni leakage-safe model veya başka bir ağırlık bu fazda terfi ettirilmedi. Faz 3 yalnızca
aktif modelin karar katmanını, iniş emniyetini, hareket ayrıştırmasını ve gözlemlenebilirliğini
güçlendirdi.

## Uygulanan kurallar

### İniş emniyeti

- UAP/UAİ kutusu görüntünün herhangi bir kenarına 1 piksel marj içinde değiyorsa
  `landing_status=0` gönderilir. Kırpılmış bir alan artık yanlışlıkla tam görünür sayılmaz.
- Alanla insan veya araç kutusunun 2B kesişimi ya da 3B frustum kesişimi varsa durum `0` olur.
- Referans eşleme sonrası bulunan bilinmeyen/referans obje kutuları için ikinci iniş kontrolü
  çalışır. Alanla kesişen referans obje de durumu `0` yapar.
- Engel yok, alan içeride ve tam görünürse durum `1` kalır.

### Duplicate kutu koruması

- Ham ONNX sonuçları confidence değerine göre sıralanır.
- Aynı sınıfta IoU `0,45` üzerinde olan kutular NMS ile tekilleştirilir.
- Farklı sınıflarda IoU `0,90` üzerinde olan neredeyse aynı kutulardan yalnızca en yüksek
  skorlu olan tutulur.
- ONNX adaptörünün arkasında ikinci, modelden bağımsız bir tekilleştirme katmanı vardır;
  kapsama oranı yüzde 90 üzerindeki aynı-sınıf iç içe kutular da elenir.

100 validation görüntüsündeki runtime probunda eşik üstü duplicate sayısı sıfırdır.

### Kamera hareketi ve araç hareketi

- Önce UAP/UAİ, sonra diğer araç-dışı landmark eşleşmeleri kullanılır.
- En az üç landmark varsa `estimateAffinePartial2D + RANSAC` ile translation, rotation ve
  scale birlikte modellenir.
- İki landmarkta kontrollü median translation, en az üç genel eşleşmede robust median
  fallback kullanılır.
- Kamera hareketini ayıracak yeterli kanıt yoksa araç `0` veya `1` diye tahmin edilmez;
  güvenli `motion_status=-1` gönderilir.
- İki karelik hysteresis ani tek-kare sıçramalarını hareket olarak işaretlemeyi engeller.

## İnsan confidence kalibrasyonu

Aktif `.pt` checkpoint, kendi validation bölümü üzerinde `conf=0.001` ile çalıştırıldı ve
insan sınıfının precision/recall eğrisi tarandı.

| Çalışma noktası | Confidence | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Önceki ortak eşik | 0,250 | 0,691 | 0,307 | 0,425 |
| Validation F1 optimum | 0,158 | 0,581 | 0,347 | 0,434 |

Yalnızca insan eşiği `0,158` yapıldı; araç, UAP ve UAİ `0,25` kaldı. 100 validation
görüntüsündeki runtime A/B probunda:

- araç: 547 → 547
- insan: 92 → 131
- UAP: 60 → 60
- UAİ: 14 → 14
- duplicate: 0 → 0

Profil `colab_result/detector_thresholds.json` dosyasındadır. Profil SHA-256 değeri
`b22b18e3b40c5de8620e49b187c7cc410e7002f363390027e5300992c2a29213` olup aktif
ONNX hash'ine ve sınıf sırasına bağlıdır. Hash veya sınıf sırası uyuşmazsa sistem fail-closed
davranır. Bu eşik mAP değerini değiştirmez; gönderilen kutuların precision/recall çalışma
noktasını değiştirir.

## RGB ve termal gözlemlenebilirlik

- Her frame metriğinde `modality` ve tespit sayısı saklanır.
- Canlı log satırı modality ve obje sayısını içerir.
- Hem bellek içi özet hem JSONL analizinde RGB, termal ve bilinmeyen modality ayrı
  `frames`, `fallback`, obje sayısı ve aşama yüzdelikleriyle raporlanır.
- Yeterlilik testi, beklenen modality altında bütün karelerin sayıldığını ayrıca kabul
  koşulu olarak doğrular.

## Runtime kanıtı

### Tam temiz oturum — termal, maksimum hız

`artifacts/qualification/phase3/clean/result.json`

| Metrik | Sonuç |
|---|---:|
| GET/POST/kabul | 2.250 / 2.250 / 2.250 |
| Fallback / timeout / restart | 0 / 0 / 0 |
| SLA ihlali | 0 |
| p95 | 378,31 ms |
| Modality kaydı | thermal 2.250 |
| Sonuç | Geçti |

### Yarışma ayarı — 1,5 FPS

`artifacts/qualification/phase3-actual-rate/`

| Senaryo | Kare | Kabul | Fallback | Timeout | SLA | p95 | Sonuç |
|---|---:|---:|---:|---:|---:|---:|---|
| Temiz RGB | 300 | 300 | 0 | 0 | 0 | 176,46 ms | Geçti |
| Kaos termal | 300 | 300 | 3 beklenen | 0 | 0 | 183,45 ms | Geçti |

Kaos koşusu bozuk görüntü, boş görüntü, boş metadata, 401, 429, 500 ve bağlantı kopmasını
gerçek HTTP yolunda uyguladı.

### Maksimum-hız kaynak baskısı — şeffaf başarısız kapı

`artifacts/qualification/phase3-final/chaos/result.json`

Bilgisayar üzerinde HürGör dışında çalışan yoğun Docker servisleri varken pacing kapatılarak
2.250 karelik RGB kaos stresi yapıldı. Protokol dayanıklılığı geçti: 2.250/2.250 POST kabul,
duplicate/sıra ihlali/deadlock/fatal hata sıfır ve p95 362,56 ms. Ancak iki inference timeout
nedeniyle fallback 29 beklenen yerine 31, model restart 2 ve SLA ihlali 3 oldu. Bu nedenle
strict test sonucu bilinçli olarak **başarısız** bırakıldı.

Bu bulgu yarışma operasyon kuralına dönüştürülmüştür: HürGör cihazında yarışma öncesi
Docker, model eğitimi, indeksleme ve başka CPU/GPU iş yükleri kapatılmalıdır. Yarışma ayarı
olan 1,5 FPS koşuları aynı makinede bu timeoutları üretmedi.

## Kalite kapıları

- `ruff check .`: geçti
- `pytest -q`: 93/93 geçti
- Aktif model preflight: geçti
- Aktif ONNX SHA değişmedi
- `.env` izni: `0600`
- İnsan eşiği yalnızca hash-bağlı profille etkin

## Model terfi kuralı

Bir sonraki model ancak aşağıdakilerin tamamını geçerse aktif `.env`ye alınacaktır:

1. Leakage-safe validation üzerinde genel ve sınıf bazlı metrikler aktif modeli aşmalı.
2. İnsan eşiği yeni modelin kendi validation eğrisinden yeniden hesaplanmalı.
3. ONNX manifesti, sınıf sırası ve SHA doğrulanmalı.
4. RGB ve termal örneklerde duplicate/landing/motion regresyonları geçmeli.
5. Temiz 2.250 karede fallback, timeout, restart ve SLA ihlali sıfır olmalı.
6. Kaos testinde yalnızca enjekte edilen bozuk/boş kareler fallback üretmeli.

Bu kapılar geçmeden `HURGOR_YOLO_ONNX_PATH`, model SHA veya threshold profili değiştirilmez.
