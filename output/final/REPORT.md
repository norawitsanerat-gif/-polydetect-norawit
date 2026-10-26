# รายงานผลการเทรน YOLO26n — Rec (Detect) และ Polygon (Segment)

## 1. ข้อมูลที่ใช้เทรน

ที่มา: ชุดย่อยของ COCO 3 คลาส (Person, Car, Bus) คัดเลือกภาพ (seed=0) ให้จำนวน object ต่อ label สมดุลและไม่น้อยกว่า 100

| ชุด | จำนวนภาพ | Person | Car | Bus |
|---|---|---|---|---|
| Train (Rec) | 95 | 136 | 137 | 133 |
| Train (Poly, polygon) | 95 | 153 | 140 | 154 |
| Val | 40 | 80 | 47 | 39 |

เกณฑ์ข้อกำหนด: ทุก label ในชุด train มี object เกิน 100

## 2. การเทรน

- โมเดลฐาน: `YOLO("yolo26n.yaml").load("yolo26n.pt")` (detect) และ `yolo26n-seg` (segment)
- `epochs=50, imgsz=640, batch=-1 (auto), freeze=10, workers=2` บน GTX 1650 4GB
- `freeze=10` = ล็อก backbone 10 ชั้น เรียนเฉพาะ head/neck
- best.pt ถูกเลือกจาก fitness ของ validation (epoch 1 ทั้ง 2 โมเดล เพราะ pretrained มี 3 คลาสนี้อยู่แล้ว)

## 3. ผลการเทรน (best.pt)

| โมเดล | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Rec (box) | 0.841 | 0.710 | **0.792** | 0.576 |
| Poly (box) | 0.807 | 0.662 | **0.745** | 0.515 |
| Poly (mask) | 0.813 | 0.645 | **0.737** | 0.451 |

## 4. ผลแยกตามคลาส (mAP50, val 40 ภาพ)

| คลาส | Rec (box) | Poly (box) | Poly (mask) |
|---|---|---|---|
| Person | 0.735 | 0.673 | 0.670 |
| Car | 0.739 | 0.725 | 0.709 |
| Bus | **0.902** | **0.833** | **0.827** |

Poly mask mAP50-95: Person 0.330, Car 0.372, Bus 0.648

## 5. ผลการ Predict (val 40 ภาพ)

| โมเดล | จำนวนภาพ | detections | Person | Car | Bus |
|---|---|---|---|---|---|
| Rec | 40 | 174 | 77 | 56 | 41 |
| Poly | 40 | 180 | 71 | 76 | 33 |

## 6. ไฟล์ที่ส่งมอบ

| path | เนื้อหา |
|---|---|
| `output/final/weights/best_rec_detect.pt` | weight Rec (5.45 MB) |
| `output/final/weights/best_poly_segment.pt` | weight Poly (6.59 MB) |
| `output/final/train_results/rec/` | results.csv/png, PR/F1/P/R curves, confusion matrix, val_batch |
| `output/final/train_results/poly/` | เหมือนข้างต้น + Mask curves |
| `output/final/predict/rec/` | ภาพทำนาย 40 ภาพ (bbox) + `predict_rec.csv` (174 แถว) |
| `output/final/predict/poly/` | ภาพทำนาย 40 ภาพ (mask) + `predict_poly.csv` (180 แถว) |
| `dataset/data/` | ภาพ + `.json` (AnyLabeling) 135 ภาพ |

## 7. วิธีใช้ weights

```python
from ultralytics import YOLO

# Rec (bbox)
model = YOLO("output/final/weights/best_rec_detect.pt")
model.predict("image.jpg")

# Poly (mask)
model = YOLO("output/final/weights/best_poly_segment.pt")
model.predict("image.jpg")
```

เว็บแอป: `run_app.bat` หรือ `uvicorn app.main:app --port 8000`

## 8. ข้อสังเกต

- Poly (segment) ให้ mask ตามขอบวัตถุจริง แต่ mAP ต่ำกว่า Rec เล็กน้อย เพราะงานยากกว่า (ต้องทำทั้ง box + mask)
- Bus ได้ mAP50 สูงสุดทั้ง 2 โมเดล เพราะวัตถุมีขนาดใหญ่และรูปร่างชัดเจน
- Val มีเพียง 40 ภาพ (166 instances) → ตัวเลข mAP อาจแกว่งได้ถ้าเปลี่ยนชุด val
- `freeze=10` ช่วยให้เทรนเร็วและลด overfit บนชุดข้อมูลขนาดเล็ก
