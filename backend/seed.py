#!/usr/bin/env python3
"""Tạo dữ liệu mẫu — chạy 1 lần trước khi khởi động server"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from models.db import init_db, SessionLocal, Employee, Guest
from pathlib import Path


def run():
    init_db()
    db = SessionLocal()

    # Nhân viên mẫu
    emps = [
        dict(name="Nguyễn Minh Tuấn", department="Kinh doanh",
             position="Trưởng phòng", phone="0901234567",
             email="tuan@cty.vn", telegram_id=""),
        dict(name="Trần Thị Lan", department="Kỹ thuật Robot",
             position="Kỹ sư", phone="0912345678",
             email="lan@cty.vn", telegram_id=""),
        dict(name="Lê Văn Hùng", department="R&D",
             position="Trưởng nhóm AI", phone="0923456789",
             email="hung@cty.vn", telegram_id=""),
        dict(name="Phạm Thu Hà", department="Ban Giám Đốc",
             position="Giám đốc", phone="0934567890",
             email="giamdoc@cty.vn", telegram_id=""),
    ]
    added = 0
    for e in emps:
        if not db.query(Employee).filter(Employee.email == e["email"]).first():
            db.add(Employee(**e))
            added += 1
    db.commit()
    print(f"✓ {added} nhân viên mẫu")

    # Khách mẫu
    guests = [
        dict(name="Nguyễn Văn Bình", company="Vingroup",
             phone="0956789012", visit_count=2),
        dict(name="Trần Thị Cúc", company="FPT Corp",
             phone="0967890123", visit_count=1),
    ]
    added = 0
    for g in guests:
        if not db.query(Guest).filter(Guest.name == g["name"]).first():
            db.add(Guest(**g))
            added += 1
    db.commit()
    print(f"✓ {added} khách mẫu")

    # File knowledge mẫu
    kb_dir = Path("data/knowledge")
    kb_dir.mkdir(parents=True, exist_ok=True)
    sample = kb_dir / "sample.txt"

    company = os.getenv("COMPANY_NAME", "Công ty AI Robot Solutions")
    desc    = os.getenv("COMPANY_DESC", "Chuyên cung cấp giải pháp Robot và AI")

    sample.write_text(f"""GIỚI THIỆU CÔNG TY
==================
{company}
{desc}

SẢN PHẨM ROBOT:
- Robot Công Nghiệp: hàn, cắt, lắp ráp. Tải trọng 10-200kg. Từ 150 triệu đồng.
- Cobot (Robot cộng tác): làm việc cùng người, an toàn. Từ 120 triệu đồng.
- Robot Kho AMR: tự hành, SLAM navigation. Từ 200 triệu đồng.
- Robot Lễ Tân AI: nhận diện khuôn mặt, giọng nói. Từ 80 triệu đồng.
- Cánh Tay Robot 7 bậc: lắp ráp linh hoạt. Từ 95 triệu đồng.

DỊCH VỤ:
- Tư vấn thiết kế hệ thống tự động hóa
- Lắp đặt, đào tạo vận hành
- Bảo trì định kỳ
- Tích hợp AI vào dây chuyền sản xuất

NHÂN SỰ:
- Phòng Kinh doanh: Nguyễn Minh Tuấn (Trưởng phòng) - 0901234567
- Phòng Kỹ thuật: Trần Thị Lan (Kỹ sư Robot) - 0912345678
- Phòng R&D: Lê Văn Hùng (Trưởng nhóm AI) - 0923456789
- Ban Giám Đốc: Phạm Thu Hà (Giám đốc) - 0934567890

LIÊN HỆ:
Hotline: 1800-76268
Email: info@airobots.vn
Giờ làm việc: Thứ 2 - Thứ 6, 8:00 - 17:30
""", encoding="utf-8")
    print(f"✓ File knowledge mẫu: {sample}")

    db.close()
    print("\n✅ Xong! Giờ chạy server:")
    print("   uvicorn main:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    run()
