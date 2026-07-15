# HürGör detector veri sözleşmesi

Sabit sınıf sırası değiştirilemez:

| ID | Sınıf | Etiketleme notu |
|---:|---|---|
| 0 | `arac` | Şartnamenin taşıt tanımına giren kara taşıtları; belirsiz bisiklet/tricycle otomatik map edilmez. |
| 1 | `insan` | Görünen insan gövdesini sıkı bounding box ile kapsar. |
| 2 | `uap` | Uçan araba park alanının görünür sınırı. |
| 3 | `uai` | Uçan ambulans iniş alanının görünür sınırı. |

Kale direği, bina, ağaç, yol çizgisi ve motor parçası bağımsız yarışma sınıfı değildir. Bir motosiklet/taşıt şartname tanımına giriyorsa tüm araç tek `arac` kutusu olarak etiketlenir; yalnız motor bölümü ayrıca etiketlenmez.

## Roboflow

1. Proje türü `Object Detection / Bounding Boxes` seçilir.
2. Yalnızca yukarıdaki dört sınıf, aynı yazım ve sırayla oluşturulur.
3. Aynı videonun komşu kareleri rastgele train/validation'a dağıtılmaz.
4. Boş/negatif kareler kutusuz bırakılır; sahte sınıf eklenmez.
5. Export biçimi `YOLOv8/YOLO11` seçilebilir; export sonrası sınıf sırası mutlaka doğrulanır.

## Yerel kalite kapısı

```bash
python3 tools/validate_yolo_labels.py /dataset/labels --report logs/labels.json
python3 tools/deduplicate_images.py /dataset/images --report logs/duplicates.json
python3 tools/build_grouped_split.py frames.jsonl artifacts/split
```

Model eğitiminde validation/test videoları train'e hiçbir şekilde girmemelidir. RGB ve termal sonuçlar, küçük nesne AP/recall ve sınıf bazlı confusion matrix ayrı raporlanır.

## Hazır dış kaynakları hazırlama

Kaynak listesi `configs/external_datasets.json` içinde tutulur. GitHub ve HuggingFace kaynakları API anahtarı istemez:

```bash
python3 tools/download_external_assets.py --skip-roboflow --skip-drive
python3 tools/build_detector_dataset.py --clean
python3 tools/validate_yolo_labels.py artifacts/datasets/hurgor_detector/labels --report logs/hurgor-detector-labels.json
python3 tools/resplit_yolo_dataset.py artifacts/datasets/hurgor_detector artifacts/datasets/hurgor_detector_grouped --clean
```

Roboflow Universe projeleri public görünse bile export indirme için genellikle API key gerekir. Key hazır olduğunda:

```bash
export ROBOFLOW_API_KEY="..."
python3 tools/download_external_assets.py --only roboflow_hepsi_aaa
python3 tools/download_external_assets.py --only roboflow_teknofest_2023_uap_uai
python3 tools/build_detector_dataset.py --clean
```

`build_detector_dataset.py`, `Vehicle/Pedestrian`, `insan/tasit/uai/uap`, `UAİ-`, `UAP-` gibi sınıf adlarını tek sözleşmeye çevirir: `arac`, `insan`, `uap`, `uai`.

`resplit_yolo_dataset.py`, aynı video veya sahneden gelen komşu kareleri tek grupta
tutar, sınıf dağılımını gözeterek yeniden böler ve train ile validation/test arasında
aynı perceptual hash'e sahip karelerden yalnızca bir tarafı korur. Eğitimde kullanılacak
ana dosya `artifacts/datasets/hurgor_detector_grouped/data.yaml` olmalıdır. Kaynak
exportların rastgele frame splitleri yalnızca ham arşiv olarak tutulur.
