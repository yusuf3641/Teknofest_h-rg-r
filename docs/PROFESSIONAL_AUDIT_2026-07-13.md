# HürGör Profesyonel Sistem Denetimi — 13 Temmuz 2026

## Karar

Sistem, çevrimiçi sunucu bağlantısı ve dayanıklı haberleşme iskeleti açısından çalışır
durumdadır. Resmî sunucuda gerçek bir kare GET ile alınmış, ONNX modelinden geçirilmiş ve
POST cevabı 523 ms uçtan uca sürede kabul edilmiştir; sunucu ilerlemesi `0 → 1` olmuştur.

Final yarışma doğruluğu için henüz tam “GO” verilmemelidir. İnsan sınıfı, termal eğitim,
ground-truth ile görsel odometri ve sızıntısız veriyle yeniden eğitim tamamlanmalıdır.

## Doğrulanan protokol

- Resmî auth, progress, classes, reference, frames, translation ve prediction endpointleri
  doğrulandı.
- Yeni kare, önceki karenin POST onayı gelmeden alınmıyor.
- Resmî POST tek JSON nesnesidir.
- `reference_predictions` zorunlu alt alanları canlı doğrulama cevabıyla bulundu:
  `reference`, `top_left_x`, `top_left_y`, `bottom_right_x`, `bottom_right_y`.
- Referans URL'si, `/reference/` cevabından oluşturulan yerel manifestten okunuyor.
- Salt-okunur bağlantı probu artık `/frames/` çağırmıyor; ilerleme `frame_index=1` olarak
  değişmeden kaldı.

## Test sonuçları

| Test | Sonuç | Önemli ölçüm |
|---|---:|---|
| Birim/contract/regresyon | 68/68 geçti | Ruff ve format temiz |
| Temiz Python 3.12 kurulumu | 68/68 geçti | ONNX preflight ve CPU provider geçti |
| Resmî sunucu gerçek çevrim | 1/1 kabul | 523 ms, fallback yok |
| Gerçek dataset görüntüleri | 120/120 POST | 1,81 gerçek çevrim FPS; 3 watchdog fallback |
| Gerçek 640x512 termal video | 40/40 POST | 0 fallback; p95 481 ms |
| Bozuk JPEG + boş GET + 40 ms ağ gecikmesi | 40/40 kabul | 5 beklenen fallback; SLA miss yok |
| Token expiry + 429 + 503 | 60/60 kabul | 46 retry; 24×401, 13×429, 9×5xx |
| Task 2 + Task 3 resmî mock | 20/20 kabul | 18 referans POST; VO p95 yaklaşık 91 ms |
| Süreç-ağacı bellek testi | 300/300 | RSS 223 MB → 124 MB; sızıntı işareti yok |

300 karelik bellek koşusunda Docker sanal makinesi CPU'yu yoğun kullandığı için 9 kare
1 saniyeyi aştı; fallback olmadı. Bu sonuç bellek denetimidir, hedef donanım benchmarkı
değildir.

## Model ve veri bulguları

Colab test sonucu genel `mAP50=0.853`, `mAP50-95=0.689` görünmektedir. Sınıf bazında insan
en zayıf noktadır (`mAP50=0.569`, `mAP50-95=0.215`). UAP ve UAİ değerleri yüksektir.

Eski grouped splitte farklı veri kaynakları altında tekrar gelen aynı/çok benzer videolar
bulundu. Bu nedenle eski mAP değerleri iyimser olabilir. Yeni
`hurgor_detector_leakage_safe` seti oluşturuldu:

- 7.337 görüntü ve 52.013 kutu; etiket hatası 0.
- 1.175 başlangıç grubu, yakın kopya bağlantılarından sonra 834 bağımsız bileşen.
- 341 grup birleştirildi.
- dHash Hamming 0–3 için train–val, train–test ve val–test çakışması 0.
- Dağılım: 5.022 train, 1.447 val, 868 test.

Mevcut `best.onnx` değiştirilmedi. Sonraki eğitim bu temiz setle yapılmalı ve yeni test
sonucu eski modelle aynı bağımsız sette karşılaştırılmalıdır.

## Bu denetimde kapatılan açıklar

- Mac M2'de kararsız CoreML fallback zinciri yerine ölçülmüş `CPUExecutionProvider` seçildi.
- Model export manifesti artık ONNX çıktı şeklini okuyarak `end2end`/`one_to_many` tipini
  doğruluyor; belirsiz biçimde fail-closed davranıyor.
- Mock sunucu gerçek görüntü klasörlerini deterministik ve sınırlı biçimde sunabiliyor.
- Mock varsayılan portu macOS Control Center çakışmasını önlemek için 8765 yapıldı.
- Bellek metriği ana süreç tepe RSS'i yerine tüm süreç ağacının anlık RSS'ini ölçüyor.
- 1080p kamera profili, 33 desenli kalibrasyon dosyasındaki gerçek intrinsics ile eşitlendi.
- SE(3) ölçeği GPS kesilmeden önceki son sağlıklı Z değerini kullanıyor.
- Referans tespitleri resmî POST'a bağlandı ve contract testi eklendi.
- Güvenli resmî probun istemeden kare rezerve etmesi engellendi.

## Kalan kritik işler

1. Temiz veri setiyle yeniden eğitim ve bağımsız karşılaştırma yapılmalı.
2. İnsan sınıfına küçük/uzak insan, zor negatif ve termal insan örnekleri eklenmeli.
3. İki termal video dört sınıfla etiketlenmeli; modalite bazlı metrik çıkarılmalı.
4. Kalibre homografi/SE(3) odometri ve ATE/RPE değerlendirme aracı tamamlandı. Gerçek
   translation CSV hâlâ eksik; gerçek MAE `< 50 m` kapısı sağlanırsa
   `HURGOR_ENABLE_EXPERIMENTAL_VO=true` yapılmalı.
5. ORB baseline, gerçek RGB↔termal referans çiftlerinde LightGlue/LoFTR ile kıyaslanmalı.
6. Yarışma cihazında CUDA/TensorRT benchmarkı ve 2.250 karelik kesintisiz soak yapılmalı.
7. Temiz Python 3.12 ortamı ve ONNX preflight doğrulandı; aktif `.venv` hâlâ 3.13 olduğu
   için yarışma öncesi doğrulanan 3.12 ortamına kontrollü geçiş yapılmalıdır.
8. Test makinesinde Docker yarışma koşusundan önce kapatılmalı; mevcut VM CPU'yu sürekli
   tüketerek watchdog zaman aşımı oluşturabiliyor.

## Kanıt dosyaları

- `artifacts/audit/preflight-final-2026-07-13.json`
- `artifacts/audit/preflight-python312.json`
- `artifacts/audit/official-one-frame.jsonl`
- `artifacts/audit/real-images-120-v1.jsonl`
- `artifacts/audit/thermal-real-40-v1.jsonl`
- `artifacts/audit/local-faults-40-v1.jsonl`
- `artifacts/audit/official-faults-60-v1.jsonl`
- `artifacts/audit/official-task23-20-v1.jsonl`
- `artifacts/audit/memory-endurance-300-v1.jsonl`
- `artifacts/audit/leakage-safe-split-audit.json`
- `artifacts/audit/leakage-safe-label-validation.json`
- `artifacts/audit/official-reference-schema-probe.json`
- `artifacts/audit/official-reference-url-validation.json`
