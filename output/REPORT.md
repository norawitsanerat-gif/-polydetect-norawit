# REPORT — YOLO Rec (bbox) vs Polygon (segment)

## สรุป
เทรน YOLOv8-nano 2 แบบบนชุดข้อมูล **COCO 2017 ใหม่ทั้งหมด** (ไม่ใช้ข้อมูลเดิม) — person/car/bus

| | Rec (detect) | Poly (segment) |
|---|---|---|
| Label format | bbox `x y w h` | polygon `x1 y1 ... xn yn` |
| Base model | yolo26n.pt | yolo26n-seg.pt |
| **mAP50** | **0.742** | Box 0.678 / Mask 0.659 |
| **mAP50-95** | **0.517** | Box 0.444 / Mask 0.401 |
| Precision | 0.743 | 0.794 |
| Recall | 0.664 | 0.564 |
| Weight size | 20 MB | 6.3 MB |

## Per-class (Rec / val 194 ภาพ)
| class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| person | 0.785 | 0.656 | 0.768 | 0.501 |
| car | 0.693 | 0.574 | 0.630 | 0.402 |
| bus | 0.750 | 0.762 | **0.829** | **0.649** |

## ข้อมูล
| | train | val |
|---|---|---|
| ภาพ | 583 | 194 |
| person | 1,730 obj | 604 |
| car | 395 obj | 108 |
| bus | 193 obj | 59 |

- ทุก label ใน train **≥100 object** ตามข้อกำหนด
- ทั้ง 2 dataset ใช้ภาพ/วัตถุชุดเดียวกัน (bbox มาจาก polygon เดียวกัน) → เทียบกันได้แฟร์

## ไฟล์ที่ส่งมอบ
| path | เนื้อหา |
|---|---|
| `output/weights/best_rec_detect.pt` | weight แบบ bbox (20 MB) |
| `output/weights/best_poly_segment.pt` | weight แบบ polygon (6.3 MB) |
| `output/train_results/rec/` | results.csv/png, PR/F1/P/R curves, confusion matrix |
| `output/train_results/poly/` | เหมือนข้างต้น + Mask curves |
| `output/predict/rec/` | ภาพทำนาย 12 ภาพ (bbox) |
| `output/predict/poly/` | ภาพทำนาย 12 ภาพ (mask) |
| `output/predict/predictions.csv` | ตาราง detections ทั้งหมด 1,211 รายการ |

## วิธีใช้
```bash
# Rec
yolo detect predict model=output/weights/best_rec_detect.pt source=path/to/img.jpg

# Poly
yolo segment predict model=output/weights/best_poly_segment.pt source=path/to/img.jpg
```

## ข้อสังเกต
- Poly (segment) ให้ mask ที่ครอบวัตถุได้แม่น แต่ **mAP ต่ำกว่า** เพราะ task ยากกว่า (ต้องทำทั้ง box + mask)
- Rec (detect) mAP50 สูงกว่า ~0.06 และ weight เล็กกว่า 3 เท่า เหมาะกับงานที่ต้องการแค่ตำแหน่ง/กล่อง
- bus ได้ mAP50 สูงสุด (0.83) เพราะวัตถุใหญ่และลักษณะชัดเจน
- ทั้ง 2 model ตรวจจับ bus/person ได้ถูกต้องในภาพตัวอย่าง `output/predict/`

## หมายเหตุการเทรน
- Rec หยุดที่ 31/100 epochs, Poly หยุดที่ 30/30 epochs (ตามคำสั่ง "ไม่ต้องเทรนโหด")
- GTX 1650 4GB, VRAM ใช้สูงสุด ~1.6GB, เวลาเทรนรวม ~37 นาที
