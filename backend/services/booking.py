"""Đặt lịch hẹn qua hội thoại tiếng Việt tự nhiên"""
import re
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict


def parse_vi_datetime(text: str) -> Optional[datetime]:
    """Phân tích ngày giờ từ tiếng Việt: 'thứ 3 tuần sau lúc 2 giờ chiều'"""
    t   = text.lower()
    now = datetime.now()
    target = now.date()
    next_week = "tuần sau" in t or "tuần tới" in t

    DAY_MAP = {
        "hôm nay": 0, "ngày mai": 1, "ngày kia": 2,
        "thứ hai": 0, "thứ 2": 0,
        "thứ ba": 1,  "thứ 3": 1,
        "thứ tư": 2,  "thứ 4": 2,
        "thứ năm": 3, "thứ 5": 3,
        "thứ sáu": 4, "thứ 6": 4,
        "thứ bảy": 5, "thứ 7": 5,
        "chủ nhật": 6,
    }
    for day, wday in DAY_MAP.items():
        if day in t:
            if day == "hôm nay":
                target = now.date()
            elif day == "ngày mai":
                target = (now + timedelta(1)).date()
            elif day == "ngày kia":
                target = (now + timedelta(2)).date()
            else:
                diff = wday - now.weekday()
                if diff <= 0 or next_week:
                    diff += 7
                target = (now + timedelta(diff)).date()
            break

    # Tìm giờ
    m = re.search(r'(\d{1,2})\s*(?:giờ|h|g)\s*(\d{0,2})', t)
    if m:
        hour   = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        if "chiều" in t and hour < 12:
            hour += 12
        if "tối" in t and hour < 18:
            hour += 12
        return datetime.combine(target, datetime.min.time().replace(hour=hour, minute=minute))
    return None


class BookingFlow:
    """State machine đặt lịch 4 bước"""

    def __init__(self, employees: List[Dict]):
        self.emps   = employees
        self.state  = "ask_employee"
        self.data: Dict = {}
        self.result: Optional[Dict] = None

    def step(self, text: str) -> Tuple[str, bool]:
        """Trả về (câu trả lời, done)"""
        t = text.lower()

        if self.state == "ask_employee":
            emp = next((e for e in self.emps
                        if e["name"].lower() in t
                        or (e.get("department", "").lower() in t and len(t) > 3)), None)
            if emp:
                self.data["employee"] = emp
                self.state = "ask_time"
                return (f"Anh/chị muốn gặp {emp['name']} lúc nào? "
                        "Ví dụ: ngày mai 2 giờ chiều.", False)
            names = ", ".join(e["name"] for e in self.emps[:5])
            return f"Anh/chị muốn gặp ai? Có thể chọn: {names}.", False

        if self.state == "ask_time":
            dt = parse_vi_datetime(text)
            if dt:
                self.data["datetime"] = dt
                self.state = "confirm"
                emp  = self.data["employee"]["name"]
                days = ["Thứ Hai","Thứ Ba","Thứ Tư","Thứ Năm","Thứ Sáu","Thứ Bảy","Chủ Nhật"]
                day  = days[dt.weekday()]
                return (f"Em sẽ đặt lịch gặp {emp} vào {day} "
                        f"ngày {dt.strftime('%d/%m/%Y')} lúc {dt.strftime('%H:%M')}. "
                        "Anh/chị xác nhận không?"), False
            return "Em chưa hiểu thời gian. Anh/chị nói rõ hơn được không? Ví dụ: thứ 3 tuần sau lúc 9 giờ sáng.", False

        if self.state == "confirm":
            if any(w in t for w in ["có", "ok", "được", "đồng ý", "xác nhận", "ừ", "đúng", "yes"]):
                self.state  = "done"
                self.result = {
                    "employee_name": self.data["employee"]["name"],
                    "employee_id":   self.data["employee"].get("id", 0),
                    "scheduled_at":  self.data["datetime"],
                }
                return "Tuyệt! Đã đặt lịch thành công. Em sẽ thông báo đến nhân viên ngay. Anh/chị cần hỗ trợ gì thêm không?", True
            if any(w in t for w in ["không", "thôi", "đổi", "hủy", "khác"]):
                self.state = "ask_time"
                return "Anh/chị muốn đổi sang thời gian khác không?", False
            return "Anh/chị xác nhận lịch hẹn chưa ạ?", False

        return "Em không hiểu. Anh/chị nói lại được không?", False
