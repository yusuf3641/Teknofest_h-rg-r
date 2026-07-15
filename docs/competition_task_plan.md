# HürGör Yarışma Görev Planı

Bu plan, sistemi TEKNOFEST 2026 Havacılıkta Yapay Zeka protokolüne göre sıraya koyar.

## P0 - Haberleşme ve kontrat

- Sunucudan her döngüde tek frame alınır.
- Aynı frame için JSON gönderilmeden yeni frame istenmez.
- `detected_objects`, `detected_translations`, `detected_undefined_objects` şeması Pydantic ile doğrulanır.
- Hata durumunda fallback JSON gönderilir.

Durum: Temel iskelet, localhost fault-injection testleri ve resmî sunucuda kabul edilmiş
tek karelik GET → inference → POST çevrimi hazır.

## P1 - Görev 1: Nesne tespiti

Resmi sınıflar:

- `0`: Taşıt (`arac`)
- `1`: İnsan (`insan`)
- `2`: UAP Alanı (`uap`)
- `3`: UAİ Alanı (`uai`)

Dataset politikası:

- `car`, `buldozer`, `truck`, `bus`, `motorcycle`, `tractor` ve benzeri araçlar `arac` sınıfına map edilir.
- `person`, `human`, `pedestrian` sınıfları `insan` sınıfına map edilir.
- `goal` / kale direği sınıfı eğitime alınmaz.
- Roboflow polygon label satırları bbox formatına çevrilir.

Eğitim komutu:

```bash
scripts/train_clean_detector.sh
```

Bu script eğitimden sonra otomatik olarak:

- `best.pt` ağırlığını üretir.
- `best.onnx` export eder.
- `best.json` manifestini yazar.
- test split değerlendirmesini `test_metrics.json` olarak kaydeder.

## P1 - İniş ve hareket durumları

- Taşıt için `motion_status` zorunludur:
  - `0`: hareketsiz
  - `1`: hareketli
- İnsan için:
  - `landing_status=-1`
  - `motion_status=-1`
- UAP/UAİ için:
  - `landing_status=1`: alan inişe uygun
  - `landing_status=0`: alan üzerinde/yanında engel var veya alan tam görünmüyor
  - `motion_status=-1`

Durum: Temel 3D frustum çakışma analizi var; boş ve bloklu alan testleri eklendi.

## P2 - Görev 2: Pozisyon kestirimi

- `gps_health_status=1` iken sunucudan gelen referans pozisyon kullanılabilir.
- `gps_health_status=0` iken görsel odometri tahmini gönderilmelidir.
- Optical-flow SE(3) modu sentetik hareket ve mock GPS kesintisi testlerinden geçmiştir;
  son sağlıklı Z yüksekliğini ölçek olarak kullanır. Gerçek konum ground-truth'u ile hata
  bütçesi ölçülene kadar çevrimiçi bağlantı profilinde varsayılan kapalıdır.

## P2 - Görev 3: Referans obje eşleme

- Sunucu `/reference/` ile oturum başında referans obje görsellerini verir.
- Bu objeler YOLO sınıfı değildir.
- Bulunan referans objeler iç modelde `detected_undefined_objects` içine `object_id` ile yazılır.
- Resmî POST'ta bunlar `reference_predictions` alanına; manifestteki `reference` URL'si ve
  dört bounding-box koordinatıyla çevrilir.
- Mevcut baseline ORB + RANSAC homography’dir.
- Resmî alt şema canlı doğrulama cevabıyla teyit edildi ve mock contract testine eklendi.
- Sonraki kalite adımı: LightGlue / SuperPoint / LoFTR tabanlı matcher adaptörü.

## P2 - Termal görüntüler

- RGB ve termal veri ayrı raporlanmalı.
- Termal frame’ler ayrıca etiketlenmeli veya RGB modeline domain-adaptation augmentasyonu eklenmelidir.
- Referans obje eşlemede termal/RGB farkı için CLAHE, grayscale ve edge-map preprocessing kullanılmalıdır.

## P3 - Dayanıklılık

- 300 frame süreç-ağacı RSS endurance testi.
- RGB ve termal mock smoke test.
- Fallback, timeout, retry, memory ve FPS metriklerinin loglanması.
