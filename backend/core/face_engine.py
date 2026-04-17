"""
Face Engine
Nhận frame từ camera Tablet (qua WebSocket) → nhận diện → callback

Logic session:
- Khi phát hiện người mới → fire event → LOCK camera (không nhận diện nữa)
- Camera chỉ mở lại khi: hội thoại kết thúc (bye), hoặc im lặng > SILENCE_TIMEOUT giây
- Trong lúc đang hội thoại camera vẫn nhận frame nhưng bỏ qua hoàn toàn
"""
import os, cv2, pickle, time, asyncio, logging
from pathlib import Path
from typing import Optional, Callable, Dict
import numpy as np

log = logging.getLogger("face")

try:
    from insightface.app import FaceAnalysis
    _INSIGHTFACE = True
except ImportError:
    _INSIGHTFACE = False

try:
    import face_recognition as _fr
    _FACEREC = True
except ImportError:
    _FACEREC = False


class FaceEngine:
    # Sau bao nhiêu giây im lặng thì coi như khách đã đi → mở lại nhận diện
    SILENCE_TIMEOUT = 120.0   # 2 phút
    # Cooldown tối thiểu giữa 2 lần nhận diện (ngay cả khi session đã kết thúc)
    MIN_COOLDOWN    = 5.0

    def __init__(self, on_event: Callable = None):
        self.on_event       = on_event
        self._known: Dict[str, dict] = {}
        self._app           = None
        self._faces_dir     = Path("data/faces")
        self._faces_dir.mkdir(parents=True, exist_ok=True)

        # ── Session state ──────────────────────────────────────────
        self._in_session    = False   # Đang trong hội thoại với khách
        self._session_start = 0.0    # Thời điểm bắt đầu session
        self._last_activity = 0.0    # Lần cuối khách nói/tương tác
        self._last_detect   = 0.0    # Lần cuối nhận diện được (tránh spam)
        self._current_key   = None   # Key của khách hiện tại

        self._init()

    def _init(self):
        if _INSIGHTFACE:
            try:
                self._app = FaceAnalysis(
                    name="buffalo_sc",
                    providers=["CPUExecutionProvider"])
                self._app.prepare(ctx_id=0, det_size=(320, 320))
                log.info("InsightFace ready")
            except Exception as e:
                log.warning(f"InsightFace failed: {e}")
        if not self._app:
            log.info("Face engine: " + ("face_recognition" if _FACEREC else "unavailable"))

    # ── Session control (gọi từ main.py) ──────────────────────────
    def session_start(self, face_key: str = None):
        """Gọi khi bắt đầu hội thoại — khoá camera nhận diện"""
        self._in_session    = True
        self._session_start = time.time()
        self._last_activity = time.time()
        self._current_key   = face_key
        log.info(f"Face session started (key={face_key}) — camera recognition LOCKED")

    def session_activity(self):
        """Gọi mỗi khi khách nói hoặc nhấn gì đó — reset silence timer"""
        self._last_activity = time.time()

    def session_end(self):
        """Gọi khi khách tạm biệt — mở lại camera nhận diện"""
        self._in_session  = False
        self._current_key = None
        self._last_detect = time.time()  # Cooldown ngắn trước khi nhận diện lại
        log.info("Face session ended — camera recognition UNLOCKED")

    @property
    def in_session(self) -> bool:
        # Tự động hết session nếu im lặng quá lâu
        if self._in_session:
            silent = time.time() - self._last_activity
            if silent > self.SILENCE_TIMEOUT:
                log.info(f"Face session auto-ended after {silent:.0f}s silence")
                self.session_end()
        return self._in_session

    # ── Load known faces ───────────────────────────────────────────
    def load_known(self, db):
        self._known = {}
        from models.db import Employee, Guest
        for e in db.query(Employee).filter(Employee.face_embedding.isnot(None)).all():
            try:
                self._known[f"emp_{e.id}"] = {
                    "id": e.id, "name": e.name,
                    "type": "employee", "department": e.department,
                    "embedding": pickle.loads(e.face_embedding)
                }
            except Exception:
                pass
        for g in db.query(Guest).filter(Guest.face_embedding.isnot(None)).all():
            try:
                self._known[f"guest_{g.id}"] = {
                    "id": g.id, "name": g.name,
                    "type": "guest",
                    "embedding": pickle.loads(g.face_embedding)
                }
            except Exception:
                pass
        log.info(f"Loaded {len(self._known)} known faces")

    # ── Core: xử lý 1 frame ───────────────────────────────────────
    async def process_frame(self, frame):
        """
        Gọi từ WebSocket handler mỗi khi Tablet gửi frame lên.
        - Nếu đang trong session → bỏ qua hoàn toàn
        - Nếu chưa có session → nhận diện, fire event, bắt đầu session
        """
        # Đang trong hội thoại → không nhận diện
        if self.in_session:
            return

        # Cooldown tối thiểu sau khi session kết thúc
        if time.time() - self._last_detect < self.MIN_COOLDOWN:
            return

        loop = asyncio.get_event_loop()
        emb, bbox = await loop.run_in_executor(None, self._embedding, frame)
        if emb is None:
            return

        now = time.time()
        key, info, score = self._match(emb)

        if key:
            # Khuôn mặt quen
            self._last_detect = now
            self.session_start(key)
            if self.on_event:
                await self.on_event({
                    "type":   "recognized",
                    "person": info,
                    "score":  score,
                    "face_key": key,
                })
        else:
            # Khuôn mặt lạ — nhưng phải thực sự thấy mặt (emb không None)
            self._last_detect = now
            self.session_start("unknown")
            if self.on_event:
                await self.on_event({"type": "unknown", "face_key": "unknown"})

    # ── Embedding & matching ───────────────────────────────────────
    def _embedding(self, frame):
        if self._app:
            faces = self._app.get(frame)
            if faces:
                return faces[0].normed_embedding.tolist(), faces[0].bbox
        if _FACEREC:
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = _fr.face_locations(rgb)
            encs = _fr.face_encodings(rgb, locs)
            if encs:
                return encs[0].tolist(), locs[0]
        return None, None

    def _match(self, emb):
        if not self._known or emb is None:
            return None, None, 0.0
        ea = np.array(emb)
        best_key, best_info, best_score = None, None, 0.0
        for key, info in self._known.items():
            kb = np.array(info["embedding"])
            s  = float(np.dot(ea, kb) / (np.linalg.norm(ea) * np.linalg.norm(kb) + 1e-9))
            if s > best_score:
                best_score, best_key, best_info = s, key, info
        thr = float(os.getenv("FACE_THRESHOLD", "0.55"))
        if best_score >= thr:
            return best_key, best_info, best_score
        return None, None, best_score

    # ── Save face ──────────────────────────────────────────────────
    def save_face(self, db, person_id: int, person_type: str, frame) -> bool:
        emb, bbox = self._embedding(frame)
        if emb is None:
            return False
        img_path = self._faces_dir / f"{person_type}_{person_id}.jpg"
        if bbox is not None:
            b = [int(v) for v in bbox[:4]]
            crop = frame[max(0,b[1]):b[3], max(0,b[0]):b[2]]
            if crop.size > 0:
                cv2.imwrite(str(img_path), crop)
        from models.db import Employee, Guest
        Model = Employee if person_type == "employee" else Guest
        obj   = db.query(Model).filter(Model.id == person_id).first()
        if obj:
            obj.face_embedding = pickle.dumps(emb)
            obj.face_image     = str(img_path)
            db.commit()
            self.load_known(db)
            return True
        return False

