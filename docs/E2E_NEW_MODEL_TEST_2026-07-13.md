# Yeni Model Uçtan Uca Yarışma Döngüsü Testi

Tarih: 13 Temmuz 2026

> **Düzeltme (13 Temmuz 2026, 18:50):** Bu rapordaki ilk E2E koşuları,
> `.env` dosyasının işaret ettiği önceki 80-epoch grouped modelin
> `colab_result/best.onnx` çıktısıyla yapılmıştır. Gerçek leakage-safe 100-epoch
> model kökteki `best.pt` dosyasıdır. Bu model ayrı ONNX paketine çıkarılmış ve
> yeniden test edilmiştir. Güncel ve karşılaştırmalı sonuçlar için
> `docs/MODEL_COMPARISON_2026-07-13.md` esas alınmalıdır.

## Sonuç

Yeni `colab_result/best.onnx` modeli yarışma boru hattına doğru şekilde bağlıdır.
GET → görüntü indirme → ONNX çıkarımı → Pydantic doğrulama → POST sırası hem
yerel hem de resmî sözleşme modunda korunmuştur. Mock sunucu bütün tamamlanan
koşulardaki POST paketlerini kabul etmiştir; deadlock veya fatal pipeline hatası
oluşmamıştır.

Sistem işlevsel olarak hazırdır, fakat Mac CPU üzerinde 1 saniyelik gecikme hedefi
henüz güvenli marjla sağlanmamaktadır. Yarışma performans kapısı NVIDIA
CUDA/TensorRT hedef cihazında tekrar doğrulanmalıdır.

## Test edilen model

- Model: `colab_result/best.onnx`
- Manifest: `colab_result/best.json`
- SHA-256: `484a766c10fe67c6177dc8ab3379c21ea4d9558a1eb95060b814402f05c1d31f`
- Sınıflar: `arac`, `insan`, `uap`, `uai`
- Girdi: 640 × 640, batch 1
- Çalışma sağlayıcısı: `CPUExecutionProvider`

## Test sonuçları

| Senaryo | Sonuç | Fallback | SLA ihlali | Uçtan uca p50 | Uçtan uca p95 | Duvar FPS |
|---|---:|---:|---:|---:|---:|---:|
| 0,9 sn watchdog, 50 gerçek kare | 50/50 kabul | 13 | 15 | 794 ms | 1.332 sn | 0,55 |
| 3,0 sn teşhis sınırı, 50 gerçek kare | 50/50 kabul | 0 | 12 | 750 ms | 1.302 sn | 1,01 |
| Bozuk/boş frame ve gecikme, 30 kare | 30/30 kabul | 4 | 4 | 668 ms | 1.382 sn | 1,06 |
| Resmî endpoint ve şema, 12 kare | 12/12 kabul | 0 | 3 | 797 ms | 1.617 sn | 0,61¹ |
| Resmî endpoint + 401/429/503 + bozuk frame | 12/12 kabul | 2 | 8 | 1.329 sn | 2.234 sn | 0,48¹ |

¹ Kısa koşuların duvar FPS değeri tek seferlik 5–8 saniyelik model ön ısıtmasını
içerir. Frame süreleri model hazırlandıktan sonra ölçülmüştür.

## Doğrulanan davranışlar

- Yeni frame, önceki frame POST ile onaylanmadan çekilmedi.
- Resmî `/auth/`, `/progress/`, `/frames/`, `/translation/`, `/reference/` ve
  `/prediction/` akışı çalıştı.
- Resmî şemada 12 kareden 70 nesne ve 3 referans eşleşmesi üretildi.
- Bozuk görüntülerde çıkarım atlandı ve şeması geçerli fallback JSON gönderildi.
- Hata enjeksiyonlu resmî turda toplam 15 HTTP retry gerçekleştirildi:
  6 adet 401, 5 adet 429 ve 4 adet 5xx.
- Hata sonrasında bütün 12 frame sunucu tarafından kabul edildi.
- Kuyruk derinlikleri sınırlı kaldı; input/output kuyruğunda birikme görülmedi.
- Kısa koşularda RSS artış eğilimi görülmedi. 2.250 frame hedef donanım testi yine
  de zorunludur.

## Test sırasında bulunan ve düzeltilen hata

0,9 saniyeyi aşan çıkarım sonrasında watchdog modeli yeniden başlatıyordu. Yeni
frame, replacement model hazır olmadan çekildiği için 3–5 saniyelik model başlangıcı
aktif frame süresine yazılıyor ve sürekli timeout zinciri oluşuyordu.

Producer artık restart edilen inference worker hazır değilse yeni GET öncesinde
hazır olmasını bekliyor. Böylece model başlangıç süresi sunucuda outstanding olan
bir frame'e yüklenmiyor. Bu davranış için regresyon testi eklendi.

## Kalan performans riskleri

1. Mac CPU'da nesne tespiti p50 yaklaşık 500–690 ms, p95 yaklaşık 1.0–1.27 sn.
2. Resmî moddaki ORB referans eşleme p50 yaklaşık 243–335 ms ekliyor.
3. 0,9 saniyelik watchdog profili 50 karenin 13'ünde fallback üretti.
4. Mevcut graceful-degradation denetleyicisi yalnızca ardışık beş yavaş frame ile
   devreye girdiği için dağınık SLA ihlallerinde ağır referans eşlemeyi kapatmıyor.
5. İnsan sınıfı eğitim metriklerinde diğer sınıflardan belirgin şekilde zayıftır;
   model performansı geliştirmesinin veri önceliği insan sınıfıdır.

## Sonraki zorunlu adımlar

1. Aynı ONNX modeli NVIDIA yarışma cihazında CUDA ve ardından TensorRT ile benchmark
   etmek.
2. Hedef cihazda resmî sözleşme moduyla en az 2.250 kare dayanıklılık testi yapmak.
3. Referans eşlemeye zaman bütçesi, görüntü küçültme, descriptor önbelleği ve hızlı
   bypass eklemek; hedef p95 en fazla 150 ms.
4. Rolling-window tabanlı SLA denetleyicisiyle dağınık yavaş frame'lerde de otomatik
   vites düşürmek.
5. Uzak/küçük insan ve termal insan örnekleriyle insan sınıfını güçlendirip yalnızca
   bu veri iyileştirmesinden sonra yeni model sürümü eğitmek.

## Kanıt dosyaları

- `artifacts/e2e/new_model_clean_v2_20260713.jsonl`
- `artifacts/e2e/new_model_cpu_diagnostic_20260713.jsonl`
- `artifacts/e2e/new_model_fault_injection_20260713.jsonl`
- `artifacts/e2e/new_model_official_contract_20260713.jsonl`
- `artifacts/e2e/new_model_official_faults_20260713.jsonl`
- Aynı isimli `.log` dosyaları
