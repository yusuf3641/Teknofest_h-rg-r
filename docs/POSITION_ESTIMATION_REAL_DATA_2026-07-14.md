# HürGör Pozisyon Kestirimi - Gerçek Veri Doğrulaması

Tarih: 14 Temmuz 2026

## Karar

GPS kesintisi motoru gerçek bir hava RGB kaydında dört farklı kesinti penceresinin
dördünde de son-konumu-tut yöntemini geçti. Projektif homografi özellikleri tam AIR
kesintisinde yönelim alanları yokken bütünleşik MAE'yi 33,60 m'ye, opsiyonel
quaternion yönelimi mevcutken 24,22 m'ye indirdi; son-konumu-tut tabanı 205,25 m'dir.
Daha önce zor olan Zurich orta penceresini de %39,28 iyileşmeyle geçti. Gerçek
termal motion-capture kaydında ise
zaman-sıralı beceri kapısı üç pencerede de adayı reddetti ve motor güvenli biçimde
son-konumu-tut davranışına döndü.

Bu nedenle yarışma yapılandırmasında `HURGOR_ENABLE_EXPERIMENTAL_VO=true` ve
`HURGOR_VO_PROJECTIVE_FEATURES=true` kullanılır; ancak hareket yalnız kronolojik
validation kapısını geçtiğinde üretilir. RGB sonucu koşullu üretim onayıdır. Termal
sonuç hatalı drift'i önlediğini kanıtlar, termal konum doğruluğunu kanıtlamaz.
Projektif profil tam 2250-kare bütünleşik koşulda da 2250/2250 cevap, sıfır
fallback/SLA/restart vermiştir. Opsiyonel yönelimli koşulda uçtan uca p95
307,41 ms, azami süre 701,94 ms ve 1800-kare kesintide 24,22 m MAE'dir.

## Resmi 2026 koşulları

Teknik şartnamenin 16-18 ve 24-25. sayfalarına göre:

- Pozisyon kestirimi yalnızca kamera görüntülerinden yapılır.
- Kamera yeryüzüne 70-90 derece bakar ve kamera parametreleri takımlarla paylaşılır.
- Akış 7,5 FPS'tir; taslak oturum 5 dakika ve 2.250 karedir.
- İlk 450 karede X/Y/Z bilgisinin sağlıklı olacağı kesindir.
- Sonraki 1.800 karede sağlık değeri bilinmeyen anda ve sürede `0` olabilir.
- Çalışma zamanında gelen taslak alanlar kare kimliği, X/Y/Z ve sağlık değeridir;
  API tablosunda IMU veya oryantasyon alanı yoktur.
- Puan, her karedeki üç boyutlu Öklid konum hatasının ortalamasıyla hesaplanır ve
  ikinci görev toplam puanın %40'ıdır.

Genel yarışma dokümanı örnek videoyla konum ve oryantasyon bilgisinin paylaşılacağını
söyler. Teknik şartnamenin canlı API tablosu ise oryantasyon alanı tanımlamaz. Uygulama
bu nedenle oryantasyon telemetrisi varsaymaz. `orientation_x/y/z/w` alanlarının
dördü birlikte ve finite gelirse normalize quaternion yaw'ı kullanılır; alanlar
yoksa görüntü-yaw yolu kesintisiz devam eder. Kısmi quaternion reddedilir.

## Uygulanan geliştirmeler

1. Harici veri setlerinin kendi `fx`, `fy`, `cx`, `cy` değerleriyle test edilmesi sağlandı.
   640x512 olması nedeniyle yanlış HürGör kamera profilinin seçilmesi engellendi.
2. Zurich Urban MAV RGB kareleri resmi distorsiyon parametreleriyle düzeltildi ve
   fotogrametrik yer gerçeği her kareye enterpole edildi.
3. IRS IROS 2021 termal ROS kaydı, resmi FLIR Boson kalibrasyonuyla düzeltildi;
   motion-capture pozları termal zaman damgalarına enterpole edildi.
4. Termal örnekleyicideki zaman damgası jitter'ı giderildi. Doğrulanmış çıktı
   6,67 yerine 10,0002 FPS oldu.
5. Değerlendirme kapısına video/CSV kare sayısı eşitliği ve P95 `< 800 ms` koşulu
   eklendi.
6. Daha fazla perspektif bilgisi kullanmak için homografi ızgara özellikleri eklendi.
   AIR ve dört Zurich penceresinde genelleştiği, IRS termalde ise güvenlik kapısı
   başarısız adayı otomatik olarak tuttuğu için yapılandırılabilir varsayılan yapıldı.
7. Opsiyonel quaternion yönelimi model, istemci, mock sunucu, CSV evaluator ve worker
   checkpoint/restore yoluna eklendi. Yaw farkı yalnız ardışık telemetri örneklerinde
   uygulanır; kesinti sonrası çift dönüş engellenir.
8. Tam 3B quaternion, ters eksen, regularizasyon ve intercept alternatifleri ölçüldü.
   Tam 3B dönüşüm yaw-only sonuca anlamlı katkı vermedi; ters eksen yüzlerce metre
   hata üretti; yalnız AIR'i iyileştirip kronolojik doğrulamayı bozan ayarlar reddedildi.

## AIR tam yarışma oturumu

Kaynak: 3422 karelik AIR UC200 hava RGB kaydı. Yarışma düzeni için ilk 450 kare
sağlıklı, sonraki 1800 kare GPS kesintili kabul edilmiştir.

| Profil | Kesinti MAE | RMSE | P95 | Final drift | Uçtan uca p95 | Protokol |
|---|---:|---:|---:|---:|---:|---|
| Projektif, yönelim yok | 33,60 m | 37,28 m | 52,87 m | 56,15 m | 239,12 ms | 2250/2250, fallback/SLA/restart 0 |
| Projektif, opsiyonel quaternion | 24,22 m | 26,09 m | 32,78 m | 27,42 m | 307,41 ms | 2250/2250, fallback/SLA/restart 0 |
| Son-konumu-tut tabanı | 205,25 m | 225,31 m | 299,78 m | 291,86 m | — | — |

Opsiyonel yönelim bütünleşik MAE'yi yönelimsiz profile göre yaklaşık `%27,90`,
son-konumu-tut tabanına göre `%88,20` azalttı. Buna rağmen 10–20 m final hedefi
geçilmedi; yönelim teknik canlı API'de garanti edilmediği için 33,60 m sonucu da
resmî risk değeri olarak korunur.

## Gerçek hava RGB sonucu

Kaynak: Zurich Urban MAV Dataset resmi örneği. Dönüştürülen paket 350 adet
1920x1080 kare ve 3,109 m yol içerir.

| Kesinti | Kare | Aday MAE | Konumu tut MAE | İyileşme | P95 | Kabul |
|---|---:|---:|---:|---:|---:|---:|
| Erken | 45-99 | 0,096 m | 0,207 m | %53,79 | 55,06 ms | Geçti |
| Orta | 90-189 | 0,262 m | 0,432 m | %39,28 | 782,56 ms* | Geçti |
| Ana | 150-319 | 0,173 m | 0,571 m | %69,73 | 667,10 ms* | Geçti |
| Geç | 180-329 | 0,171 m | 0,354 m | %51,79 | 587,84 ms* | Geçti |

Dört koşuda da kalibrasyon kesintiden önce hazırdı, GPS dönüşündeki yeniden
sabitleme hatası `0,0 m` ve P95 süre 800 ms altındaydı. `*` işaretli süre koşuları
Mac'in başka ağır süreçlerle paylaşıldığı anda alınmıştır; doğruluk deterministiktir,
hedef cihaz süre yeterliliği değildir. İzole erken koşu 55,06 ms, AIR tam koşu
18,07 ms p95 vermiştir.

## Gerçek termal stres sonucu

Kaynak: IRS Radar Thermal Visual Inertial Datasets IROS 2021 `mocap_dark_fast`.
Paket 710 adet 640x512 termal kare, 10 FPS ve 85,93 m kapalı döngü hareket içerir.

| Kesinti | Aday MAE | Konumu tut MAE | P95 | Yeniden sabitleme | Kabul |
|---|---:|---:|---:|---:|---:|
| Erken, 90-239 | 2,543 m (hold) | 2,543 m | 60,94 ms | 0,0 m | Kapı reddetti |
| Orta, 200-499 | 2,032 m (hold) | 2,032 m | 52,94 ms | 0,0 m | Kapı reddetti |
| Geç, 400-679 | 1,682 m (hold) | 1,682 m | 49,01 ms | 0,0 m | Kapı reddetti |

Ham projektif aday termal validation'da son-konumu-tut tabanını güvenilir biçimde
geçemedi. Üretim motoru bu nedenle üç pencerede de `navigation_ready=false` tuttu;
tabloda görülen aday değerleri bilinçli güvenli hold çıktısıdır. Son pencereye göre
parametre seçilip yanıltıcı bir başarı iddiası yapılmadı.

IRS kaydı karanlık ve hızlı motion-capture stres testidir; yarışmadaki 70-90 derece
alt-görüş koşulunun birebir eşi değildir. Buna rağmen termal genelleme eksikliğini erken
görmek için değerli bir negatif testtir.

## Tekrarlama komutları

RGB ana pencere:

```bash
PYTHONPATH=src .venv/bin/python tools/evaluate_odometry.py \
  artifacts/odometry/zurich-real/zurich-mav-undistorted.mp4 \
  artifacts/odometry/zurich-real/translation.csv \
  --dropout-start 150 --dropout-end 320 --recovery-frames 3 \
  --camera-fx 893.39010814 --camera-fy 898.32648616 \
  --camera-cx 951.1310043 --camera-cy 555.13350077 \
  --camera-modality rgb --ignore-camera-profile \
  --max-runtime-p95-ms 800 --require-gate \
  --output artifacts/odometry/zurich-real/evaluation-primary.json
```

Termal temel pencere:

```bash
PYTHONPATH=src .venv/bin/python tools/evaluate_odometry.py \
  artifacts/odometry/irs-thermal-real/irs-mocap_dark_fast-thermal.avi \
  artifacts/odometry/irs-thermal-real/translation.csv \
  --dropout-start 200 --dropout-end 500 --recovery-frames 3 \
  --camera-fx 404.98 --camera-fy 405.06 \
  --camera-cx 319.05 --camera-cy 251.84 \
  --camera-modality thermal --ignore-camera-profile \
  --max-runtime-p95-ms 800 \
  --output artifacts/odometry/irs-thermal-real/evaluation-middle.json
```

## Sonraki kabul kapısı

1. Resmi örnek videonun kareleri, translation dosyası ve paylaşılan kamera parametreleri
   tek bir sabit manifestle projeye alınacak.
2. İlk 450 kare yalnızca kalibrasyon için kullanılacak; sonraki 1.800 karede en az erken,
   orta, geç ve uzun kesinti pencereleri sınanacak.
3. RGB ve termal oturumlar ayrı raporlanacak. Her spektrumda aday MAE konumu-tut MAE'den
   düşük, yeniden sabitleme hatası sıfır ve P95 800 ms altında olmalı.
4. Kapı geçmeyen akışın pose üretmediği ve hold'a döndüğü ayrıca doğrulanacak;
   yarışma `.env` dosyasında VO açık olsa bile bu güvenlik şartı kaldırılmayacak.
5. Resmî örnekte yönelim alanları varsa opsiyonel yol, yoksa görüntü-yaw yolu aynı
   2250-kare kabul kapısından geçirilecek. Final hedef her iki durumda da MAE `<20 m`dir.

## Kanıt dosyaları

- `artifacts/odometry/zurich-real/manifest.json`
- `artifacts/odometry/air-uc200-synthetic/evaluation-full-projective-v3.json`
- `artifacts/odometry/air-uc200-synthetic/evaluation-full-projective-orientation-v4.json`
- `artifacts/qualification/air-vo-v8-projective-full/clean/result.json`
- `artifacts/qualification/air-vo-v9-orientation-600/clean/result.json`
- `artifacts/qualification/air-vo-v9-orientation-full/clean/result.json`
- `artifacts/odometry/zurich-real/evaluation-zurich-{early,middle,primary,late}-projective-v3*.json`
- `artifacts/odometry/irs-thermal-real/manifest.json`
- `artifacts/odometry/irs-thermal-real/evaluation-{early,middle,late}.json`
- `artifacts/odometry/irs-thermal-real/evaluation-{early,middle,late}-projective-v3.json`
