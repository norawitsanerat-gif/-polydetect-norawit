# PolyDetect — YOLO26n Rec + Polygon (Person / Car / Bus)

งานนี้เทรน 2 โมเดลจากภาพชุดเดียวกัน แล้วนำโมเดล Polygon ไปใช้ในเว็บแอป

- **Rec (Detect)** — `yolo26n.pt` → `custom2.yaml`
- **Polygon (Segment)** — `yolo26n-seg.pt` → `custom3.yaml`

## โครงสร้างโปรเจกต์

```
projectyolo/
├── dataset/
│   ├── data/           ← ภาพ + .json (AnyLabeling)   train 95 / val 40
│   ├── rectrain/       ← YOLO detect  (95 train / 40 val)
│   └── polytrain/      ← YOLO segment (95 train / 40 val)
├── yolo2tr.ipynb       ← เทรน 2 โมเดล (freeze=10, 50 epochs)
├── yolo1.ipynb         ← predict โมเดล Rec
├── yolo2.ipynb         ← predict โมเดล Polygon
├── custom2.yaml        ← config Rec
├── custom3.yaml        ← config Polygon
├── runs/               ← ผลการเทรน (rec_stage1, poly_stage1)
├── output/final/       ← weight + ผลเทรน + ผล predict + REPORT.md
├── app/                ← เว็บแอป (FastAPI, ใช้ best_poly_segment.pt)
├── requirements.txt
└── render.yaml
```

## ผลการเทรน (best epoch)

| โมเดล | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Rec (box) | 0.841 | 0.710 | **0.792** | 0.576 |
| Poly (box) | 0.807 | 0.662 | **0.745** | 0.515 |
| Poly (mask) | 0.813 | — | **0.737** | 0.451 |

รายละเอียดทั้งหมด: `output/final/REPORT.md`

## เตรียมสภาพแวดล้อม (ครั้งแรก / เครื่องใหม่)

ดับเบิลคลิก `setup.bat` — จะสร้าง `.venv`, ติดตั้งไลบรารี และลงทะเบียน Jupyter kernel ให้อัตโนมัติ

| ไฟล์ | ใช้เมื่อ |
|---|---|
| `requirements-train.txt` | เครื่องนี้ (CUDA 12.4 + GTX 1650) — ใช้เทรน/predict |
| `requirements.txt` | ตอน deploy บน Render (CPU ล้วน) |

## วิธีรันเว็บแอป

ดับเบิลคลิก `run_app.bat` หรือ

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

เปิด http://localhost:8000

## วิธีรัน notebook

เปิดไฟล์ `.ipynb` ใน VS Code → **Select Kernel** → เลือก **Python (projectyolo .venv)**

ลำดับการรัน: `yolo2tr.ipynb` (เทรน) → `yolo1.ipynb` (predict Rec) → `yolo2.ipynb` (predict Poly)

## Dataset

- ภาพจาก **COCO 2017** (โจทย์กำหนดให้เลือกภาพเอง) คัดเฉพาะ 3 คลาส
- ชุดเทรน 95 ภาพ / val 40 ภาพ — ทุกคลาสมี object ≥ 100 ต่อ label (ตัวเลขใน `output/final/REPORT.md`)
- label ถูกเปิดตรวจ/ปรับใน **AnyLabeling** — ไฟล์ `.json` อยู่ใน `dataset/data/` (พร้อมใช้)
