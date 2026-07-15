# Termal Veri Durumu

Bu not 2026-07-12 yerel dosya envanterine göre hazırlanmıştır.

## Sonuç

Projede ayrı ve güçlü, etiketli bir termal eğitim seti henüz yok. Buna karşılık
`termal_4dk.mp4` ve `termal_veri_1_dk.mp4` gerçek 640x512 termal videoları yerelde mevcut.

Mevcut görüntülerde `thermal`, `termal` veya `ir` kelimesi geçen örnekler var; ancak dosyalar
Pillow tarafından `RGB` olarak okunuyor. Bu, görüntülerin termal görünümlü veya isimlendirilmiş
olabileceğini, fakat eğitim tarafında ayrı kalibre edilmiş termal modalite olarak ele alınmadığını
gösterir.

## Sayısal durum

- `Hurgor-YapayZeka-Gorev1`: 935 görüntü, tamamı `1920x1080`, tamamı `RGB`
- Dosya adında termal/thermal/ir geçen yerel Roboflow görüntüsü: 18
- `artifacts/datasets/hurgor_detector_grouped`: 7.232 görüntü
- Grouped dataset içinde termal isimli görüntü: 36 civarı, tamamı `RGB` modunda
- `termal_veri_1_dk.mp4`: 1.801 frame, 29.97 FPS, 640x512
- `termal_4dk.mp4`: 2.640 frame, 29.97 FPS, 640x512
- Gerçek termal runtime testi: 40/40 POST, 0 fallback, p95 uçtan uca yaklaşık 481 ms

## Teknik karar

Şu an termal için ayrı YOLO modeli eğitmek doğru değil; veri miktarı ve etiket kalitesi yetersiz.

Doğru kısa vadeli çözüm:

1. RGB modeli ana model olarak kullanılır.
2. Mock server termal modda ayrıca çalıştırılır.
3. Runtime preprocessing termal benzeri görüntülerde grayscale/CLAHE ile stabilize edilir.
4. Referans obje eşlemede ORB için CLAHE zaten kullanılır.

Doğru orta vadeli çözüm:

1. Yarışma komitesinin verdiği termal videolar ayrıca frame'lere ayrılır.
2. Aynı 4 sınıfla etiketlenir:
   - `arac`
   - `insan`
   - `uap`
   - `uai`
3. RGB ve termal görüntüler dataset manifestinde ayrı `modality` olarak tutulur.
4. Eğitimde RGB + termal karışık kullanılır.
5. Validasyon raporu RGB ve termal için ayrı çıkarılır.

## Eksik veri

Termal güçlendirme için takımdan gerekenler:

- Orijinal termal video dosyaları
- Varsa termal referans obje görselleri
- Termal frame'ler üzerinde en azından UAP/UAİ ve insan/araç etiketleri

Bu dosyalar gelmeden termal tarafında yapılabilecek en güvenli iş mock test ve preprocessing
sağlamlığıdır; gerçek model kalitesi için veri gerekir.
