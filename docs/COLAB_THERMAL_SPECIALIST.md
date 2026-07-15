# HürGör termal uzman YOLO26n — Colab eğitimi

Bu akış yalnız `arac` ve `insan` sınıflarına sahip ilk termal uzman adayını üretir.
Çıkan model dört sınıflı ana yarışma modelinin yerine otomatik olarak alınmaz. Önce
bağımsız termal testte mevcut modelden daha iyi olduğu kanıtlanır; daha sonra RGB,
UAP ve UAİ verileriyle dört sınıflı karma model hazırlanır.

## 1. Doğrulanmış paketi Drive'a koy

Roboflow'dan gelen ham çıktı doğrudan kullanılmaz. Projede 9 problemli etiket
onarılmış, sınıf sırası sabitlenmiş ve yollar Colab'a uygun hale getirilmiş şu paketi
kullan:

```text
hurgor_thermal_v1_yolo26.zip
```

ZIP'i Google Drive `MyDrive` köküne yükle. Drive'da açmana gerek yoktur. ZIP şu yolu
oluşturmalıdır:

```text
/content/drive/MyDrive/hurgor_thermal_v1_yolo26.zip
```

## 2. Colab GPU ve paketler

Colab'da `Runtime > Change runtime type > GPU` seç. A100 varsa `BATCH=32`, T4/L4
için `BATCH=16` kullan.

```python
!nvidia-smi
!pip install -q ultralytics==8.4.93 onnx onnxslim pyyaml
```

## 3. Drive'ı bağla ve girişleri kontrol et

```python
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path

ZIP_PATH = Path("/content/drive/MyDrive/hurgor_thermal_v1_yolo26.zip")
BASE_MODEL = Path("/content/drive/MyDrive/hurgor_colab_results/final_model/best.pt")
PROJECT_DIR = Path("/content/drive/MyDrive/hurgor_training")

print("ZIP:", ZIP_PATH, ZIP_PATH.exists())
print("Başlangıç modeli:", BASE_MODEL, BASE_MODEL.exists())

if not ZIP_PATH.is_file():
    raise FileNotFoundError(f"Termal ZIP bulunamadı: {ZIP_PATH}")
if not BASE_MODEL.is_file():
    candidates = sorted(Path("/content/drive/MyDrive").rglob("best.pt"))
    print("Drive'daki best.pt adayları:")
    for candidate in candidates:
        print(candidate)
    raise FileNotFoundError("BASE_MODEL yolunu yukarıdaki doğru best.pt ile değiştir")
```

## 4. ZIP bütünlüğü ve veri seti doğrulaması

Bu hücre ZIP'i geçici Colab diskine açar, sınıfları `0: arac`, `1: insan` sırasına
çevirir, bozuk kutuları ve aynı görüntünün farklı splitlerde bulunmasını reddeder.
Roboflow'daki İngilizce `car/vehicle` ve `person/human` adları desteklenir.

```python
import hashlib
import shutil
import zipfile
from collections import Counter, defaultdict

import yaml

WORK_DIR = Path("/content/hurgor_thermal_v1")
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir(parents=True)

with zipfile.ZipFile(ZIP_PATH) as archive:
    broken = archive.testzip()
    if broken is not None:
        raise RuntimeError(f"Bozuk ZIP üyesi: {broken}")
    archive.extractall(WORK_DIR)

yaml_candidates = list(WORK_DIR.rglob("data.yaml"))
if len(yaml_candidates) != 1:
    raise RuntimeError(f"Tek data.yaml bekleniyordu: {yaml_candidates}")
SOURCE_YAML = yaml_candidates[0]
source = yaml.safe_load(SOURCE_YAML.read_text())

raw_names = source.get("names")
if isinstance(raw_names, dict):
    names = [str(raw_names[key]) for key in sorted(raw_names, key=lambda value: int(value))]
else:
    names = [str(value) for value in raw_names or []]

aliases = {
    "arac": 0,
    "araç": 0,
    "car": 0,
    "vehicle": 0,
    "insan": 1,
    "person": 1,
    "human": 1,
}
normalized = [aliases.get(name.strip().lower()) for name in names]
if len(names) != 2 or None in normalized or set(normalized) != {0, 1}:
    raise RuntimeError(
        f"Yalnız araç ve insan bekleniyordu; data.yaml names={names}. Eğitim durduruldu."
    )
old_to_new = {old: new for old, new in enumerate(normalized)}

split_paths = {}
for split in ("train", "val", "test"):
    raw_path = source.get(split)
    if raw_path is None:
        if split == "test":
            continue
        raise RuntimeError(f"data.yaml içinde {split} yolu yok")
    image_dir = (SOURCE_YAML.parent / str(raw_path)).resolve()
    if not image_dir.is_dir() and str(raw_path).startswith("../"):
        image_dir = (SOURCE_YAML.parent / str(raw_path)[3:]).resolve()
    if not image_dir.is_dir():
        raise RuntimeError(f"{split} görüntü klasörü bulunamadı: {image_dir}")
    split_paths[split] = image_dir

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
counts = defaultdict(Counter)
hash_owners = {}

for split, image_dir in split_paths.items():
    label_candidates = (
        image_dir.parent / "labels",
        SOURCE_YAML.parent / "labels" / split,
        SOURCE_YAML.parent / ("valid" if split == "val" else split) / "labels",
    )
    label_dir = next((path for path in label_candidates if path.is_dir()), None)
    if label_dir is None:
        raise RuntimeError(f"{split} label klasörü bulunamadı: {label_candidates}")
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in image_extensions)
    if not images:
        raise RuntimeError(f"{split} split boş")
    counts[split]["images"] = len(images)

    for image_path in images:
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        previous = hash_owners.get(digest)
        if previous is not None and previous[0] != split:
            raise RuntimeError(
                f"Aynı görüntü iki splitte: {previous[1]} ve {image_path}"
            )
        hash_owners[digest] = (split, image_path)

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise RuntimeError(f"Görüntünün etiketi yok: {image_path}")
        rewritten = []
        for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise RuntimeError(f"Geçersiz YOLO satırı: {label_path}:{line_number}")
            old_class = int(parts[0])
            if old_class not in old_to_new:
                raise RuntimeError(f"Tanımsız sınıf: {label_path}:{line_number}")
            box = [float(value) for value in parts[1:]]
            if not all(0.0 <= value <= 1.0 for value in box) or box[2] <= 0 or box[3] <= 0:
                raise RuntimeError(f"Geçersiz normalize kutu: {label_path}:{line_number}")
            new_class = old_to_new[old_class]
            counts[split]["arac" if new_class == 0 else "insan"] += 1
            rewritten.append(" ".join([str(new_class), *parts[1:]]))
        label_path.write_text("\n".join(rewritten) + ("\n" if rewritten else ""))

DATA_YAML = WORK_DIR / "data_hurgor.yaml"
normalized_yaml = {
    "train": str(split_paths["train"]),
    "val": str(split_paths["val"]),
    "nc": 2,
    "names": ["arac", "insan"],
}
if "test" in split_paths:
    normalized_yaml["test"] = str(split_paths["test"])
DATA_YAML.write_text(yaml.safe_dump(normalized_yaml, sort_keys=False, allow_unicode=True))

print("Kaynak sınıflar:", names)
print("Yeni sınıf sırası: 0=arac, 1=insan")
print("Split sayımları:")
for split, values in counts.items():
    print(split, dict(values))
print("Eğitim YAML:", DATA_YAML)
```

Komşu video karelerinin farklı splitlere rastgele dağılması exact hash kontrolüyle
yakalanamaz. En güvenli ayrım kaynak video/sahne bazlıdır. Roboflow split'i kareleri
rastgele böldüyse sonuç iyimser olabilir ve model seçiminde bağımsız HIT-UAV testi
zorunludur.

## 5. Eğitimi başlat

Bu model iki sınıflı bir uzmandır. Aktif dört sınıflı `best.pt` yalnız transfer
öğrenme başlangıcıdır; Ultralytics iki sınıflı yeni detection head oluşturur.

```python
from datetime import datetime
from ultralytics import YOLO

BATCH = 32  # A100; T4/L4 için 16 yap
RUN_NAME = "hurgor-yolo26n-thermal-specialist-v1-" + datetime.now().strftime("%Y%m%d-%H%M")

model = YOLO(str(BASE_MODEL))
model.train(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=640,
    batch=BATCH,
    device=0,
    workers=2,
    patience=20,
    seed=2026,
    deterministic=True,
    amp=True,
    cache=True,
    project=str(PROJECT_DIR),
    name=RUN_NAME,
    plots=True,
    save_period=10,
    close_mosaic=10,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.15,
    degrees=5.0,
    translate=0.10,
    scale=0.30,
    fliplr=0.50,
    flipud=0.50,
    mosaic=0.50,
    mixup=0.05,
)

RUN_DIR = PROJECT_DIR / RUN_NAME
print("Sonuç klasörü:", RUN_DIR)
```

## 6. Bağımsız test ve ONNX export

```python
from ultralytics import YOLO

BEST_PT = RUN_DIR / "weights/best.pt"
if not BEST_PT.is_file():
    raise FileNotFoundError(BEST_PT)

best = YOLO(str(BEST_PT))
split = "test" if "test" in split_paths else "val"
test_result = best.val(
    data=str(DATA_YAML),
    split=split,
    imgsz=640,
    batch=BATCH,
    device=0,
    workers=2,
    plots=True,
)

print("split:", split)
print("mAP50:", float(test_result.box.map50))
print("mAP50-95:", float(test_result.box.map))

onnx_path = best.export(
    format="onnx",
    imgsz=640,
    dynamic=True,
    simplify=True,
    opset=17,
    device="cpu",
)
print("PT:", BEST_PT)
print("ONNX:", onnx_path)
```

## 7. Kabul kararı

Mevcut dört sınıflı modelin HIT-UAV termal sonuçları araç için `mAP50=0,344`, insan
için `mAP50=0,032` idi. Yeni aday aynı bağımsız test üzerinde mutlaka bu değerleri
geçmelidir. İlk pratik hedefler:

- termal araç `mAP50 >= 0,60`,
- termal insan `mAP50 >= 0,40`,
- termal insan recall `>= 0,40`,
- yanlış/eksik sınıf veya split kaçağı olmaması.

Roboflow test metriği tek başına yeterli değildir. Model yerel projeye alındıktan
sonra HIT-UAV ve gerçek 2026 termal videoda tekrar ölçülmeden ana runtime'a bağlanmaz.
