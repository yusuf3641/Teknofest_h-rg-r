# HürGör YOLO eğitimini Google Colab'a taşıma

Bu doküman, yerelde çok yavaş kalan YOLO eğitimini Colab GPU üzerinde çalıştırmak için kullanılır.

## Neden Colab?

Mac MPS üzerinde yapılan denemede `hurgor-yolo26n-clean-v2-2` eğitimi yaklaşık 12 saatte sadece 10 epoch tamamladı. Bu hız final model için yeterli değil. Aynı iş Colab GPU üzerinde çok daha pratik tamamlanır.

10 epoch ara modelin test sonucu:

| Sınıf | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| arac | 0.890 | 0.706 | 0.794 | 0.558 |
| insan | 0.677 | 0.366 | 0.469 | 0.176 |
| uap | 0.617 | 0.962 | 0.919 | 0.769 |
| uai | 0.909 | 0.952 | 0.970 | 0.858 |
| toplam | 0.773 | 0.746 | 0.788 | 0.590 |

Yorum:

- `uap` ve `uai` iyi durumda.
- `arac` kullanılabilir ama geliştirilebilir.
- `insan` zayıf; daha fazla doğru etiket, daha yüksek çözünürlük veya daha uzun eğitim gerekir.

## 1. Yerelde Colab paketini oluştur

Proje kökünde:

```bash
scripts/package_colab_training.sh
```

Bu komut şunu üretir:

```text
artifacts/colab/hurgor_colab_training.zip
```

Zip içinde dataset, eğitim araçları, export/evaluate araçları ve pretrained YOLO checkpoint bulunur.

## 2. Colab ayarı

Colab'da:

1. Runtime > Change runtime type
2. Hardware accelerator: GPU
3. Aşağıdaki hücreleri sırayla çalıştır

## 3. Colab hücreleri

### Paketleri kur

```python
!nvidia-smi
!pip install -q ultralytics onnx onnxslim pyyaml
```

### Zip'i yükle ve aç

Sol dosya panelinden `hurgor_colab_training.zip` dosyasını yükle.

```python
!unzip -q hurgor_colab_training.zip -d /content/hurgor_train
%cd /content/hurgor_train
```

### Eğitimi başlat

Önerilen ilk final eğitim:

```python
!python tools/train_detector.py \
  --data artifacts/datasets/hurgor_detector_grouped/data.yaml \
  --model artifacts/pretrained/yolo26n.pt \
  --project artifacts/training \
  --name hurgor-yolo26n-colab-v1 \
  --epochs 80 \
  --image-size 640 \
  --batch 16 \
  --device 0 \
  --workers 2 \
  --patience 20 \
  --no-cache \
  --plots
```

Eğer GPU belleği yetmezse:

```python
# batch 16 yerine 8 kullan
```

Eğer eğitim hızlıysa ve daha iyi doğruluk istenirse:

```python
# image-size 640 yerine 960 denenebilir.
# batch GPU belleğine göre 4 veya 8'e düşürülür.
```

### ONNX export

```python
!python tools/export_detector.py \
  artifacts/training/hurgor-yolo26n-colab-v1/weights/best.pt \
  --target onnx \
  --image-size 640 \
  --device cpu \
  --manifest artifacts/training/hurgor-yolo26n-colab-v1/weights/best.json
```

### Test metriklerini çıkar

```python
!python tools/evaluate_detector.py \
  artifacts/training/hurgor-yolo26n-colab-v1/weights/best.pt \
  --data artifacts/datasets/hurgor_detector_grouped/data.yaml \
  --split test \
  --image-size 640 \
  --device 0 \
  --batch 16 \
  --workers 2 \
  --no-plots \
  --output artifacts/training/hurgor-yolo26n-colab-v1/test_metrics.json
```

### Sonuçları indir

```python
from google.colab import files

base = "artifacts/training/hurgor-yolo26n-colab-v1"
for path in [
    f"{base}/weights/best.pt",
    f"{base}/weights/best.onnx",
    f"{base}/weights/best.json",
    f"{base}/results.csv",
    f"{base}/test_metrics.json",
]:
    files.download(path)
```

## 4. Yerel projeye geri koyma

Colab'dan indirilen dosyaları yerelde şu klasöre koy:

```text
artifacts/training/hurgor-yolo26n-colab-v1/
```

Yarışma pipeline'ında kullanılacak dosya:

```text
artifacts/training/hurgor-yolo26n-colab-v1/weights/best.onnx
```

## 5. Karar kriteri

Model yarışma pipeline'ına alınmadan önce minimum hedef:

- Toplam `mAP50 >= 0.80`
- `uap` ve `uai` için `mAP50 >= 0.90`
- `insan` için recall mevcut 0.36 seviyesinden anlamlı şekilde yukarı çıkmalı
- ONNX inference mock pipeline içinde 1 FPS sınırını rahat geçmeli

Eğer `insan` hâlâ düşük kalırsa, sorun modelden çok veri/etiket tarafındadır: insan etiketleri artırılmalı, küçük/uzak insan örnekleri özellikle eklenmelidir.
