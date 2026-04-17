"""
Face Engine
Nhận frame từ camera Tablet (qua WebSocket) → nhận diện → callback
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
    COOLDOWN = 30.0

    def __init__(self, on_event: Callable = None):
        self.on_event  = on_event
        self._known: Dict[str, dict] = {}
        self._seen:  Dict[str, float] = {}
        self._app    = None
        self._faces_dir = Path("data/faces")
        self._faces_dir.mkdir(parents=True, exist_ok=True)
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
        ea   = np.array(emb)
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

    async def process_frame(self, frame):
        """Xử lý 1 frame — gọi từ WebSocket handler"""
        loop = asyncio.get_event_loop()
        emb, bbox = await loop.run_in_executor(None, self._embedding, frame)
        if emb is None:
            return
        now = time.time()
        key, info, score = self._match(emb)
        if key and now - self._seen.get(key, 0) > self.COOLDOWN:
            self._seen[key] = now
            if self.on_event:
                await self.on_event({"type": "recognized", "person": info, "score": score})
        elif key is None and emb is not None:
            if now - self._seen.get("unknown", 0) > self.COOLDOWN:
                self._seen["unknown"] = now
                if self.on_event:
                    await self.on_event({"type": "unknown"})

    def save_face(self, db, person_id: int, person_type: str, frame) -> bool:
        emb, bbox = self._embedding(frame)
        if emb is None:
            return False
        # Lưu ảnh
        img_path = self._faces_dir / f"{person_type}_{person_id}.jpg"
        if bbox is not None:
            b = [int(v) for v in bbox[:4]]
            crop = frame[max(0,b[1]):b[3], max(0,b[0]):b[2]]
            if crop.size > 0:
                cv2.imwrite(str(img_path), crop)
        # Lưu DB
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
