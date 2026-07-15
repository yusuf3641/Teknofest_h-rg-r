# HürGör Runtime Yeterlilik Raporu — 13 Temmuz 2026

## Sonuç

`2026-07-13-runtime-v1` sürümü, aktif YOLO26n modeli ve gerçek termal video ile iki ayrı 2.250 karelik uçtan uca oturumda kabul koşullarını geçti.

| Kabul koşulu | Temiz oturum | Hata enjeksiyonlu oturum |
|---|---:|---:|
| Mantıksal frame GET | 2.250 | 2.250 |
| Şeması doğrulanmış prediction payload | 2.250 | 2.250 |
| Sunucunun kabul ettiği prediction | 2.250 | 2.250 |
| Duplicate kabul | 0 | 0 |
| Sıra/protokol ihlali | 0 | 0 |
| Deadlock | 0 | 0 |
| Fatal hata | 0 | 0 |
| Inference timeout | 0 | 0 |
| Model restart | 0 | 0 |
| SLA ihlali | 0 | 0 |
| Uçtan uca p95 | 156,27 ms | 160,91 ms |
| Uçtan uca en kötü süre | 283,53 ms | 189,13 ms |
| Gerçek duvar saati hızı | 1,486 FPS | 1,484 FPS |
| Fallback | 0 | 29 (tamamı beklenen) |

800 ms p95 sınırı her iki oturumda da geniş güvenlik payıyla sağlandı. Temiz oturumda fallback ve SLA ihlali sıfır kaldı.

## Tek GET / tek POST kanıtı

Temiz oturumda ağ seviyesinde de tam olarak 2.250 POST isteği yapıldı ve 2.250'si kabul edildi. Her POST'tan sonra bir sonraki frame GET'i açıldı; bekleyen frame, duplicate POST ve sıra ihlali oluşmadı.

Hata enjeksiyonlu oturumda 401/429/500 nedeniyle taşıma katmanı 45 ek POST denemesi yaptı. Buna rağmen yalnızca 2.250 geçerli uygulama payload'ı işlendi, 2.250 farklı frame kabul edildi ve duplicate kabul sayısı sıfır kaldı. Yeniden deneme, aynı mantıksal POST'un güvenli tekrarından ibarettir.

## Hata toleransı kapsamı

Kaos oturumunda aşağıdaki arızalar gerçekten üretildi:

- 19 bozuk görüntü
- 10 sıfır bayt/boş görüntü
- 14 boş veya geçersiz frame metadata yanıtı
- 32 adet HTTP 401
- 54 adet HTTP 429
- 39 adet HTTP 500
- 36 adet bağlantı kopması

Bozuk ve boş görüntüler için toplam 29 fallback JSON gönderildi. Boş metadata ve ağ/HTTP hataları kontrollü retry ile toparlandı. Akış kapanmadı, kilitlenmedi ve frame sırası değişmedi.

## Aktif model ve çalışma ayarları

- Model: YOLO26n, ONNX, 640 px, dinamik batch 1
- Sınıflar: `arac`, `insan`, `uap`, `uai`
- Aktif dosya: `colab_result/best.onnx`
- SHA-256: `484a766c10fe67c6177dc8ab3379c21ea4d9558a1eb95060b814402f05c1d31f`
- Sağlayıcı: `CPUExecutionProvider`
- ONNX thread: intra=1, inter=1
- OpenCV thread: 1
- Hedef hız: 1,5 FPS
- SLA: 1,0 saniye
- Inference timeout: 0,9 saniye
- Queue `maxsize`: 3
- NoOp detector: kapalı
- Inference watchdog process: açık

Model dosyası ve metadata salt okunur yapıldı. Aktif `.env` izni `0600` olarak sınırlandı. Model hash kilidi preflight sırasında doğrulanıyor; yanlış veya değişmiş modelle resmi çalışma başlatılamıyor.

## Yarışma sunucusu doğrulaması

Gerçek yarışma sunucusuna frame çekmeden salt-okunur preflight uygulandı:

- Kimlik doğrulama: başarılı
- `progress`: HTTP 200
- `reference`: HTTP 200
- `classes`: HTTP 200

Frame GET yapılmadığı için yarışma oturumunda cevapsız frame bırakılmadı. Tam 2.250 kare testleri, gerçek HTTP trafiği kullanan ve resmi endpoint/JSON sözleşmesini uygulayan yerel mock sunucuda yapıldı.

## Dondurulmuş sürüm

Yerel, git tarafından izlenmeyen güvenli paket:

`artifacts/releases/2026-07-13-runtime-v1/`

Paket; aktif ONNX/PT modeli, model metadata'sı, `0600` izinli yarışma `.env` yedeği, kaynak kod arşivi, temiz/kaos ham metrikleri, loglar ve SHA-256 manifestini içerir. Manifest dosyası hiçbir kullanıcı adı, parola veya token içermez.

## Son doğrulamalar

- `ruff check .`: başarılı
- `pytest -q`: 73/73 başarılı
- Yerel runtime preflight: başarılı
- Gerçek yarışma ağı preflight: başarılı
- Temiz 2.250 kare yeterlilik testi: başarılı
- Kaos 2.250 kare yeterlilik testi: başarılı

Not: RSS temiz koşuda yaklaşık 397 MB'den 215 MB'ye, kaos koşusunda 397 MB'den 244 MB'ye indi; bu oturumlarda kalıcı bellek büyümesi gözlenmedi.
