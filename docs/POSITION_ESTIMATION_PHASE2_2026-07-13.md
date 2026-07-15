# HürGör Pozisyon Kestirimi — Faz 2

Tarih: 13 Temmuz 2026

> 14 Temmuz gerçek veri sonuçları ve güncel etkinleştirme kararı için
> [`POSITION_ESTIMATION_REAL_DATA_2026-07-14.md`](POSITION_ESTIMATION_REAL_DATA_2026-07-14.md)
> belgesine bakın. Bu belge ilk sentetik fazın tarihsel kaydıdır.

## Sonuç

GPS sağlığı `0` olduğunda yalnızca son konumu tekrarlayan güvenli baseline korunurken,
ayrı bir kalibre görsel odometri motoru geliştirildi. Motor çalışan YOLO modelini veya
resmî haberleşme akışını değiştirmez. `HURGOR_ENABLE_EXPERIMENTAL_VO` varsayılan olarak
`false` kalır; gerçek video + gerçek translation CSV kapısı geçilmeden yarışma modunda
açılmayacaktır.

Gerçek yarışma translation CSV'si Desktop, Downloads ve proje içinde bulunamadı. Bu
sebeple 103 m gerçek hatanın düşürüldüğü henüz iddia edilemez. Hazırlanan değerlendirme
aracı veri geldiği anda bu iddiayı sayısal ve tekrarlanabilir biçimde sınayacaktır.

## Uygulanan mimari

1. Görüntü, mevcut RGB/termal kamera profiliyle distorsiyondan arındırılır ve CLAHE ile
   düşük kontrasta karşı güçlendirilir.
2. Shi–Tomasi özellikleri iki yönlü Lucas–Kanade ile takip edilir. İleri–geri takip hatası
   yüksek noktalar elenir.
3. RANSAC homografi sahnedeki aykırı hareketleri ayırır. Benzerlik dönüşümü ile global
   kamera dönüşü ve ölçek değişimi, görüntü düzlemindeki translation'dan ayrıştırılır.
4. Birikmiş görüntü yaw açısı kullanılarak kamera eksenleri dünya yönüne stabilize edilir.
5. GPS'in sağlıklı olduğu ardışık karelerde görüntü hareketi ile gerçek X/Y/Z farkı
   eşleştirilir. Kalite ağırlıklı, Huber dayanımlı regresyon kamera eksenlerini ve
   piksel/ölçek → metre dönüşüm matrisini çevrimiçi öğrenir.
6. GPS kesildiğinde öğrenilmiş metrik delta 4×4 `SE(3)` pose üzerinde biriktirilir.
7. GPS geri geldiğinde translation doğrudan gerçek değere sabitlenir; odometri drift'i
   sonraki karelere taşınmaz.

## Güvenlik davranışları

- Kalibrasyon için varsayılan en az 20 geçerli hareket çifti gerekir.
- Homografi en az 24 inlier ve %45 inlier oranı ister.
- Reprojection hatası varsayılan 3 px üzerinde ise hareket kullanılmaz.
- Tahmini adım, sağlıklı GPS hareket dağılımından çıkarılan dayanıklı sınırı veya 20 m
  mutlak sınırı aşarsa reddedilir.
- Özellik bulunamazsa son hız sınırsız sürdürülmez; her karede `0.85` katsayısıyla
  azaltılan kontrollü fallback uygulanır ve sonunda konum tutulur.
- Video veya oturum değiştiğinde görüntü geçmişi, kalibrasyon ve `SE(3)` pose sıfırlanır.
- Bilinmeyen çözünürlükte yanlış kamera profili seçilmez; açıkça verilen intrinsics
  kullanılır.

## Doğrulama sonuçları

Deterministik fakat gerçekten hareket ettirilmiş görüntülerden oluşan 240 karelik,
7,5 FPS video ve bağımsız translation CSV ile 100–219 arası GPS kesintisi sınandı.

| Test | Aday MAE | Son-konumu-tut MAE | Odometri P95 | Yeniden sabitleme |
|---|---:|---:|---:|---:|
| Sentetik RGB | 0,028 m | 46,070 m | 22,66 ms | 0,0 m hata |
| Sentetik termal | 0,680 m | 46,070 m | 15,92 ms | 0,0 m hata |

Bu sonuçlar görüntü işleme, eksen kalibrasyonu, dönüş dengelemesi, `SE(3)` birikimi ve
CSV değerlendirme zincirinin birlikte çalıştığını kanıtlayan regresyon sonuçlarıdır.
Gerçek saha doğruluğunu veya yarışma puanını temsil etmez.

Aktif YOLO26n modeli ve resmî mock GET/POST sözleşmesiyle ayrıca iki adet 240 karelik
producer/worker/consumer oturumu yapıldı. Özellik bayrağı yalnızca süreç ortamında geçici
olarak açıldı; kalıcı `.env` değiştirilmedi.

| Runtime koşusu | Kabul | Fallback | SLA ihlali | Uçtan uca P95 | Odometri P95 |
|---|---:|---:|---:|---:|---:|
| Temiz | 240/240 | 0 | 0 | 192,76 ms | 26,44 ms |
| Kaos | 240/240 | 3 (beklenen) | 0 | 220,88 ms | 33,02 ms |

Kaos koşusu bozuk görüntü, sıfır bayt görüntü, geçersiz metadata, 401, 429, 500 ve
bağlantı kopmasını içerdi; deadlock, fatal hata, sıra ihlali veya reddedilen POST oluşmadı.

Otomatik test kapsamı ayrıca şunları doğrular:

- dönüş ve ölçeğin translation'dan ayrılması;
- üç eksenli metrik dönüşümün öğrenilmesi;
- 30 dereceden fazla kamera yaw değişiminde eksen stabilizasyonu;
- özellik kaybında sıçramayan ve azalan fallback;
- GPS dönüşünde tam yeniden sabitleme;
- gerçek termal çözünürlükte video + CSV uçtan uca kabul kapısı;
- özellik bayrağı kapalıyken eski güvenli davranışın korunması.

## Gerçek veri geldiğinde çalıştırılacak kapı

CSV'nin her video karesi için aynı sırada bir satırı ve şu kolonları olmalıdır:

```text
translation_x,translation_y,translation_z,frame_numbers
```

Örnek komut:

```bash
PYTHONPATH=src .venv/bin/python tools/evaluate_odometry.py \
  /mutlak/yol/yarışma-video.mp4 \
  /mutlak/yol/translation.csv \
  --stride 1 \
  --dropout-start 450 \
  --dropout-end 900 \
  --recovery-frames 3 \
  --target-mae-m 50 \
  --reference-error-m 103 \
  --require-gate \
  --output artifacts/odometry/real-csv-evaluation.json
```

`--require-gate`, aşağıdaki koşullardan biri sağlanmazsa komutu hata koduyla bitirir:

- kesinti başlamadan kalibrasyon hazır;
- aday MAE son-konumu-tut MAE'den düşük;
- aday MAE 103 m'den düşük;
- aday MAE 50 m'den düşük;
- GPS dönüşünde yeniden sabitleme hatası tolerans içinde.

Rapor ayrıca eksen bazlı MAE, ATE RMSE, translation RPE, p50/p95/maksimum hata,
son drift, odometri süresi, öğrenilen metrik dönüşüm matrisi ve fallback tanılarını içerir.

## Kanıt dosyaları

- `artifacts/qualification/odometry-v2/evaluation-rgb.json`
- `artifacts/qualification/odometry-v2/evaluation-thermal.json`
- `artifacts/qualification/odometry-v2/clean/result.json`
- `artifacts/qualification/odometry-v2/chaos/result.json`

## Etkinleştirme kararı

İlk başarı kapısı gerçek veride MAE `< 50 m` ve son-konumu-tut baseline'ına karşı net
iyileşmedir. Bu sağlanınca farklı başlangıç/bitiş noktalarında en az üç kesinti penceresi
ayrı ayrı sınanmalıdır. Ardından `HURGOR_ENABLE_EXPERIMENTAL_VO=true` yalnızca kontrollü
test `.env` dosyasında açılmalı ve 2.250 kare temiz + kaos oturumları yeniden yapılmalıdır.

Final 10–20 m hedefi için gerçek veride kalan hata kaynakları; kamera pitch/roll
telemetrisi, düzlemsel olmayan sahne, irtifa değişimi, düşük termal doku ve hareketli
nesne oranı üzerinden analiz edilmelidir. Bu hedef gerçek CSV sonucu görülmeden garanti
edilemez.
