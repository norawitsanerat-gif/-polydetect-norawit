# dataset/data — ภาพ + label สำหรับ AnyLabeling

```
dataset/data/
├── train/   95 ภาพ + .json (447 shapes: Person 153, Car 140, Bus 154)
└── val/     40 ภาพ + .json (166 shapes: Person 80,  Car 47,  Bus 39)
```

## วิธีเปิดดู

1. เปิด AnyLabeling
2. `File → Open Dir` → เลือก `dataset\data\train`
3. ถ้าถาม output dir → เลือก **Same as image**
4. ไล่ดูได้ด้วยปุ่ม `D` (ถัดไป) / `A` (ก่อนหน้า)

## หมายเหตุ

- ชื่อคลาส: Person, Car, Bus
- พิกัดใน `.json` เป็น pixel ตามมาตรฐาน AnyLabeling
- label ชุดนี้ตรงกับ `dataset/rectrain` และ `dataset/polytrain` (ใช้เทรนแล้ว)
