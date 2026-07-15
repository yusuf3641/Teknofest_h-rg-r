# Eski–Yeni Model Karşılaştırması

Tarih: 13 Temmuz 2026

## Kısa karar

Yeni leakage-safe model **genel doğrulukta henüz daha iyi değildir**. Buna karşılık
daha hızlıdır, UAİ sınıfında daha iyidir ve eğitim/test ayrımı güvenilir olduğu için
ölçümleri yarışma genellemesini değerlendirmeye daha uygundur.

Varsayılan `.env` bilinçli olarak değiştirilmemiştir; çalışma sistemi hâlâ eski
`colab_result/best.onnx` modelini kullanmaktadır. Yeni model ayrı aday paketi olarak
saklanmıştır.

## Model kimlikleri

| Model | Eğitim | Veri | Runtime ONNX |
|---|---:|---|---|
| Eski grouped v1 | 80 epoch | `hurgor_detector_grouped` | `colab_result/best.onnx` |
| Yeni leakage-safe v2 | 100 epoch | `hurgor_detector_leakage_safe` | `artifacts/models/hurgor_yolo26n_leakage_safe_v2/best.onnx` |

Yeni ONNX SHA-256:
`20ba9e698d1d9c9a79812fdcde8b273c64d7c942051c66bfa1223612b6089000`

## Veri sızıntısı bulgusu

Leakage-safe test kümesi 868 görüntüdür. Bu görüntülerin:

- 548'i eski modelin train kümesinde,
- 26'sı eski modelin validation kümesinde,
- toplam 574'ü (%66,1) eski geliştirme sürecinde görülmüştür.

Bu nedenle 868 karenin tamamındaki eski model sonucu bağımsız başarı ölçümü olarak
kullanılamaz. Tam testte eski modelin mAP50 değeri 0,860, yeni modelinki 0,794
görünse de bu karşılaştırma eski model lehine sızıntılıdır.

## Daha adil 294 görüntülük karşılaştırma

Eski train ve validation kümeleriyle dosya bazında çakışan 574 görüntü çıkarıldı.
Kalan 294 görüntü iki modele aynı ayarlarla verildi.

| Metrik | Eski | Yeni | Yeni farkı |
|---|---:|---:|---:|
| Precision | 0,831 | 0,839 | +0,008 |
| Recall | 0,780 | 0,757 | -0,023 |
| mAP50 | 0,828 | 0,799 | -0,029 |
| mAP50-95 | 0,667 | 0,654 | -0,013 |

Sınıf bazında:

| Sınıf | Eski mAP50 | Yeni mAP50 | Eski mAP50-95 | Yeni mAP50-95 |
|---|---:|---:|---:|---:|
| Araç | 0,870 | 0,835 | 0,631 | 0,609 |
| İnsan | 0,548 | 0,463 | 0,215 | 0,170 |
| UAP | 0,976 | 0,974 | 0,940 | 0,937 |
| UAİ | 0,920 | 0,924 | 0,882 | 0,899 |

Yeni model biraz daha yüksek precision ile daha seçici davranıyor ancak recall
kaybediyor. En büyük kayıp insan sınıfında; UAİ yeni modelde daha iyi.

## Runtime hız karşılaştırması

Apple M2 `CPUExecutionProvider`, 640 × 640, aynı 60 görüntü:

| Metrik | Eski ONNX | Yeni ONNX |
|---|---:|---:|
| Ortalama detector süresi | 42,8 ms | 35,1 ms |
| p50 | 37,9 ms | 30,9 ms |
| p95 | 72,6 ms | 51,5 ms |
| p99 | 120,1 ms | 64,5 ms |

Yeni ONNX ortalamada yaklaşık %18 daha hızlıdır ve p99 kuyruğu yaklaşık %46 daha
düşüktür.

Resmî mock endpointleri, ORB referans eşleme ve 0,9 saniye watchdog açık 100 kare:

| Metrik | Eski | Yeni |
|---|---:|---:|
| Sunucu kabulü | 100/100 | 100/100 |
| Duvar FPS | 7,07 | 7,26 |
| Uçtan uca p50 | 104 ms | 103 ms |
| Uçtan uca p95 | 193 ms | 164 ms |
| Fallback | 0 | 0 |
| SLA ihlali | 0 | 0 |
| Toplam tespit | 567 | 527 |

Her iki model de hız sınırını rahat geçmiştir. Yeni modelin daha az tespit üretmesi,
daha yüksek precision ve daha düşük recall bulgusuyla uyumludur.

## İnsan sınıfındaki temel eksik

Temiz veri setinde 53.013 kutu vardır:

- Araç: 34.306 (%64,7)
- İnsan: 14.334 (%27,0)
- UAP: 1.773 (%3,3)
- UAİ: 2.600 (%4,9)

Sorun yalnızca insan örneği sayısı değildir; hedefler çok küçüktür. Train kümesinde
insan kutusunun 640 girişte medyan boyutu yaklaşık **8 × 18 piksel**dir:

- İnsan kutularının %95,35'inde en az bir kenar 16 pikselden küçüktür.
- İnsan kutularının %99,38'inde en az bir kenar 32 pikselden küçüktür.
- İnsan mAP50-95 değeri yeni modelde yalnızca 0,142'dir.

Nano model ve 640 çözünürlük bu kadar küçük insanları yeterli özellik hücresiyle
temsil edememektedir.

## Önerilen v3 çalışması

1. Leakage-safe test kümesini dondur; eğitim veya model seçimi için bir daha açma.
2. Colab A100'de iki aday eğit:
   - YOLO26s, 640 piksel,
   - YOLO26n veya YOLO26s, 960 piksel.
3. İnsan içeren gerçek drone/termal görüntülerini artır; özellikle 16–64 piksel arası
   temiz kutular ve zor negatifler ekle.
4. İnsan ağırlıklı görüntüler için crop/tiling tabanlı eğitim örnekleri üret; kutuları
   kör biçimde büyütme veya aynı kareyi aşırı çoğaltma.
5. Validation üzerinde insan confidence eşiğini ayrıca kalibre et; genel 0,25 eşiğini
   doğrudan bütün sınıflar için düşürme.
6. Adayları önce validation ile seç, yalnızca final adayı 868 karelik testte bir kez
   ölç.
7. Kabul kapıları:
   - genel mAP50-95 > 0,67,
   - insan mAP50 > 0,55,
   - insan mAP50-95 > 0,22,
   - resmî E2E p95 < 800 ms,
   - 2.250 karede fallback/SLA ihlali = 0.

## Kanıt dosyaları

- `artifacts/evaluation/new_leakage_safe_v2_test_20260713.json`
- `artifacts/evaluation/old_grouped_v1_test_20260713.json`
- `artifacts/evaluation/new_model_unseen294_20260713.json`
- `artifacts/evaluation/old_model_unseen294_20260713.json`
- `artifacts/evaluation/new_leakage_safe_v2_onnx_test_20260713.json`
- `artifacts/evaluation/benchmark_new_leakage_safe_v2_cpu_20260713.json`
- `artifacts/evaluation/benchmark_old_grouped_v1_cpu_20260713.json`
- `artifacts/e2e/new_leakage_safe_v2_official_repeat_100_20260713.jsonl`
- `artifacts/e2e/old_grouped_v1_official_comparison_100_20260713.jsonl`
