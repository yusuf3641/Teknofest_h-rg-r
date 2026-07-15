# HürGör detector model kapısı

## Adaylar

- Champion adayı: YOLO26s, 960 px, dört HürGör sınıfı.
- Düşük gecikme fallback adayı: YOLO26n, 640 px.
- Challenger: RF-DETR Small; yalnız aynı video-gruplu holdout üzerinde champion'dan iyi ise seçilir.

YOLO26'nın one-to-one başlığı `(N, 300, 6)` NMS'siz çıktı, `end2end=False`
one-to-many başlığı ise `(N, nc+4, predictions)` çıktı verir. Runtime iki formatı
manifest üzerinden ayırır; shape'e bakarak sessiz tahmin yapmaz.

Birincil kaynaklar:

- https://docs.ultralytics.com/models/yolo26/
- https://docs.ultralytics.com/guides/end2end-detection/
- https://docs.ultralytics.com/modes/export/
- https://github.com/roboflow/rf-detr

RF-DETR'nin yayımlanmış T4/TensorRT gecikmeleri M2/CoreML sonucu değildir. M2 seçimi
yalnız `tools/benchmark_detector.py` sustained ölçümü ve gerçek HürGör holdout mAP/recall
sonucuyla yapılır.

## Kabul kapısı

1. Dataset split video/scene bazlıdır ve group kaçağı yoktur.
2. Manifest sınıf sırası tam `arac, insan, uap, uai` olmalıdır.
3. Model SHA-256 değeri manifest ve `.env` ile eşleşmelidir.
4. RGB, termal, small-object ve sınıf bazlı metrikler ayrı üretilmelidir.
5. 640/960/1280 ile full-frame ve gerekirse tile varyantları p95 uçtan uca bütçede karşılaştırılır.
6. PyTorch-export parity golden image setinde doğrulanmadan yarışma modeli değiştirilmez.
7. Model dosyası yoksa resmî koşu fail-closed olur.
