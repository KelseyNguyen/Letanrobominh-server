"""
Aria AI Lễ Tân — Main Application
Stack: FastAPI + Groq API + ChromaDB + gTTS
"""
import os, sys, json, base64, asyncio, logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from contextlib import asynccontextmanager

# Load .env trước mọi import khác
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("aria")

# Local imports
from models.db import init_db, get_db, SessionLocal, Employee, Guest, KnowledgeDoc, Appointment, Conversation
from core.ai_engine import KnowledgeBase, GroqLLM, GroqSTT, TTSEngine, ChatSession
from core.face_engine import FaceEngine
from services.telegram import Telegram
from services.booking import BookingFlow

# ── Globals ─────────────────────────────────────────────────────────
kb:       Optional[KnowledgeBase] = None
llm:      Optional[GroqLLM]       = None
stt:      Optional[GroqSTT]       = None
tts:      Optional[TTSEngine]     = None
face:     Optional[FaceEngine]    = None
telegram: Optional[Telegram]      = None

tablet_ws: Dict[str, WebSocket]  = {}
admin_ws:  Dict[str, WebSocket]  = {}
sessions:  Dict[str, ChatSession] = {}
bookings:  Dict[str, BookingFlow] = {}


# ── Startup / Shutdown ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global kb, llm, stt, tts, face, telegram
    log.info("=== Aria starting ===")

    # Tạo thư mục cần thiết
    for d in ["data/knowledge", "data/faces", "data/tts_cache",
              "data/chroma_db", "data/voice_samples"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    init_db()
    log.info("DB ready")

    kb   = KnowledgeBase()
    llm  = GroqLLM(kb)
    stt  = GroqSTT()
    tts  = TTSEngine()
    log.info("AI engines ready")

    # Face engine
    face = FaceEngine(on_event=_on_face)
    db   = SessionLocal()
    try:
        face.load_known(db)
    finally:
        db.close()

    # Camera local (nếu cài đặt)
    cam = int(os.getenv("CAMERA_INDEX", "-1"))
    if cam >= 0:
        # Camera local chỉ dùng khi chạy local, không dùng trên Railway
        log.info(f"Camera index {cam} configured (local mode)")

    # Telegram
    telegram = Telegram()
    telegram.on_reply  = _on_telegram_reply
    telegram.broadcast = _broadcast_tablet
    await telegram.start()

    # Ingest file knowledge mẫu nếu có
    sample = Path("data/knowledge/sample.txt")
    if sample.exists() and kb._col and kb._col.count() == 0:
        n = await kb.ingest_file(str(sample), "sample")
        log.info(f"Ingested sample knowledge: {n} chunks")

    log.info("=== Aria ready ===")
    yield

    await telegram.stop()
    log.info("=== Aria stopped ===")


app = FastAPI(title="Aria AI Receptionist", version="2.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

Path("static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Face event handler ───────────────────────────────────────────────
async def _on_face(evt: dict):
    etype  = evt.get("type")
    person = evt.get("person")

    if etype == "recognized" and person:
        name   = person["name"]
        ptype  = person["type"]
        text   = (f"Chào {name}! Chào mừng trở lại." if ptype == "employee"
                  else f"Chào {name}! Rất vui được gặp lại.")
        sid    = f"face_{person['id']}"
        if sid not in sessions:
            sessions[sid] = ChatSession(
                guest_id   = person["id"] if ptype == "guest" else None,
                guest_name = name         if ptype == "guest" else None)
        await _say_and_broadcast(text, sid)

    elif etype == "unknown":
        text = "Xin chào! Em là Aria, lễ tân của công ty. Anh/chị cần em hỗ trợ gì không ạ?"
        sid  = f"new_{int(datetime.now().timestamp())}"
        sessions[sid] = ChatSession()
        await _broadcast_tablet({"type": "face_unknown", "greeting": text, "session_key": sid})
        await _say_and_broadcast(text, sid)


async def _on_telegram_reply(reply_text: str, emp_name: str):
    msg   = f"Nhân viên {emp_name} nhắn: {reply_text}"
    audio = await tts.speak(msg)
    await _broadcast_tablet({"type": "assistant_text", "text": msg})
    if audio:
        await _broadcast_tablet({"type": "audio",
                                  "data": base64.b64encode(audio).decode()})


# ── WebSocket: Tablet ────────────────────────────────────────────────
@app.websocket("/ws/tablet/{cid}")
async def ws_tablet(ws: WebSocket, cid: str):
    await ws.accept()
    tablet_ws[cid] = ws
    log.info(f"Tablet connected: {cid}")
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            await _handle_tablet(msg, cid, ws)
    except WebSocketDisconnect:
        tablet_ws.pop(cid, None)
        log.info(f"Tablet disconnected: {cid}")
    except Exception as e:
        log.error(f"Tablet WS error: {e}")
        tablet_ws.pop(cid, None)


async def _handle_tablet(msg: dict, cid: str, ws: WebSocket):
    t   = msg.get("type", "")
    sid = msg.get("session_key", cid)

    # ── Audio từ mic ────────────────────────────────────────────────
    if t == "audio_chunk":
        b64 = msg.get("data", "")
        if not b64:
            return
        text = await stt.transcribe(base64.b64decode(b64))
        if not text.strip():
            return

        await ws.send_text(json.dumps({"type": "user_text", "text": text}))

        session = sessions.setdefault(sid, ChatSession())
        result  = await llm.respond(session, text)

        await ws.send_text(json.dumps({
            "type":        "assistant_text",
            "text":        result["text"],
            "intent":      result.get("intent"),
            "robot_model": result.get("robot_model"),
            "guest_name":  result.get("guest_name"),
        }))

        if result.get("robot_model"):
            await ws.send_text(json.dumps(
                {"type": "show_catalog", "model": result["robot_model"]}))

        audio = await tts.speak(result["text"])
        if audio:
            await ws.send_text(json.dumps(
                {"type": "audio", "data": base64.b64encode(audio).decode()}))

        # Telegram notify nếu khách muốn gặp nhân viên
        if result.get("intent") == "contacting":
            asyncio.create_task(_notify_employee(session, result["text"]))

        # Booking flow
        if result.get("intent") == "booking":
            asyncio.create_task(_booking_step(sid, text, session, ws))

        asyncio.create_task(_save_conv(session, text, result["text"]))

    # ── Frame ảnh từ camera Tablet ──────────────────────────────────
    elif t == "camera_frame":
        b64 = msg.get("data", "")
        if b64 and face:
            import cv2, numpy as np
            arr   = np.frombuffer(base64.b64decode(b64), np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                asyncio.create_task(face.process_frame(frame))

    # ── Text message (không dùng mic) ───────────────────────────────
    elif t == "text_message":
        text = msg.get("text", "").strip()
        if not text:
            return
        session = sessions.setdefault(sid, ChatSession())
        result  = await llm.respond(session, text)

        await ws.send_text(json.dumps({
            "type":        "assistant_text",
            "text":        result["text"],
            "intent":      result.get("intent"),
            "robot_model": result.get("robot_model"),
        }))
        if result.get("robot_model"):
            await ws.send_text(json.dumps(
                {"type": "show_catalog", "model": result["robot_model"]}))
        audio = await tts.speak(result["text"])
        if audio:
            await ws.send_text(json.dumps(
                {"type": "audio", "data": base64.b64encode(audio).decode()}))

    # ── Admin say (phát qua loa từ Telegram hoặc Admin UI) ──────────
    elif t == "admin_say":
        text = msg.get("text", "")
        if text:
            await ws.send_text(json.dumps({"type": "assistant_text", "text": text}))
            audio = await tts.speak(text)
            if audio:
                await ws.send_text(json.dumps(
                    {"type": "audio", "data": base64.b64encode(audio).decode()}))

    elif t == "ping":
        await ws.send_text(json.dumps({"type": "pong"}))


# ── WebSocket: Admin ─────────────────────────────────────────────────
@app.websocket("/ws/admin/{cid}")
async def ws_admin(ws: WebSocket, cid: str):
    await ws.accept()
    admin_ws[cid] = ws
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "ping":
                await ws.send_text(json.dumps(
                    {"type": "pong", "sessions": len(sessions)}))
            elif msg.get("type") == "reload_faces":
                db = SessionLocal()
                try:
                    face.load_known(db)
                    await ws.send_text(json.dumps(
                        {"type": "faces_reloaded",
                         "count": len(face._known)}))
                finally:
                    db.close()
    except WebSocketDisconnect:
        admin_ws.pop(cid, None)
    except Exception:
        admin_ws.pop(cid, None)


# ── Helpers ──────────────────────────────────────────────────────────
async def _broadcast_tablet(msg: dict):
    dead = []
    data = json.dumps(msg)
    for cid, ws in tablet_ws.items():
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(cid)
    for cid in dead:
        tablet_ws.pop(cid, None)


async def _say_and_broadcast(text: str, sid: str):
    await _broadcast_tablet({"type": "assistant_text", "text": text, "session_key": sid})
    if tts:
        audio = await tts.speak(text)
        if audio:
            await _broadcast_tablet(
                {"type": "audio", "data": base64.b64encode(audio).decode()})


async def _notify_employee(session: ChatSession, context: str):
    if not telegram or not telegram.token:
        return
    db = SessionLocal()
    try:
        emp = db.query(Employee).filter(
            Employee.is_active == True,
            Employee.telegram_id != "").first()
        if emp:
            await telegram.notify(
                emp.telegram_id, emp.name,
                session.guest_name or "Khách", context[:100])
    except Exception as e:
        log.error(f"Notify employee error: {e}")
    finally:
        db.close()


async def _booking_step(sid: str, text: str, session: ChatSession, ws: WebSocket):
    if sid not in bookings:
        db   = SessionLocal()
        emps = [{"id": e.id, "name": e.name, "department": e.department or ""}
                for e in db.query(Employee).filter(Employee.is_active == True).all()]
        db.close()
        bookings[sid] = BookingFlow(emps)
        # Bước đầu tiên
        flow = bookings[sid]
        reply, done = flow.step(text)
    else:
        flow = bookings[sid]
        reply, done = flow.step(text)

    if reply:
        audio = await tts.speak(reply)
        await ws.send_text(json.dumps({"type": "assistant_text", "text": reply}))
        if audio:
            await ws.send_text(json.dumps(
                {"type": "audio", "data": base64.b64encode(audio).decode()}))

    if done:
        if flow.result:
            db   = SessionLocal()
            appt = Appointment(
                guest_name    = session.guest_name or "",
                employee_name = flow.result["employee_name"],
                employee_id   = flow.result["employee_id"],
                scheduled_at  = flow.result["scheduled_at"])
            db.add(appt); db.commit(); db.close()
            # Thông báo Telegram
            if telegram and telegram.token:
                db2  = SessionLocal()
                emp  = db2.query(Employee).filter(
                    Employee.id == flow.result["employee_id"]).first()
                if emp and emp.telegram_id:
                    from services.telegram import Telegram
                    dt  = flow.result["scheduled_at"]
                    await telegram.send(
                        emp.telegram_id,
                        f"📅 Lịch hẹn mới:\n"
                        f"Khách: {session.guest_name or 'Khách'}\n"
                        f"Lúc: {dt.strftime('%H:%M %d/%m/%Y')}")
                db2.close()
        bookings.pop(sid, None)


async def _save_conv(session: ChatSession, user: str, assistant: str):
    db = SessionLocal()
    try:
        for role, content in [("user", user), ("assistant", assistant)]:
            db.add(Conversation(
                guest_id=session.guest_id or 0,
                role=role, content=content))
        db.commit()
    except Exception as e:
        log.error(f"Save conv error: {e}")
    finally:
        db.close()


async def _broadcast_admin(msg: dict):
    data = json.dumps(msg)
    dead = []
    for cid, ws in admin_ws.items():
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(cid)
    for cid in dead:
        admin_ws.pop(cid, None)


# ── REST API ─────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status":  "ok",
        "groq":    bool(os.getenv("GROQ_API_KEY")),
        "tablets": len(tablet_ws),
        "sessions":len(sessions),
    }

@app.get("/api/stats")
async def stats(db=Depends(get_db)):
    return {
        "employees":    db.query(Employee).filter(Employee.is_active == True).count(),
        "guests":       db.query(Guest).count(),
        "docs":         db.query(KnowledgeDoc).filter(KnowledgeDoc.status == "ready").count(),
        "appointments": db.query(Appointment).count(),
        "sessions":     len(sessions),
        "tablets":      len(tablet_ws),
    }

# Chat API (test, không cần mic)
@app.post("/api/chat")
async def chat_api(data: dict):
    text = data.get("text", "").strip()
    sid  = data.get("session_key", "demo")
    if not text:
        raise HTTPException(400, "text is required")
    if not llm:
        raise HTTPException(503, "LLM not ready")
    session = sessions.setdefault(sid, ChatSession())
    return await llm.respond(session, text)

# Employees
@app.get("/api/employees")
async def list_employees(db=Depends(get_db)):
    rows = db.query(Employee).filter(Employee.is_active == True).all()
    return [{"id": e.id, "name": e.name, "department": e.department,
             "position": e.position, "phone": e.phone,
             "email": e.email, "telegram_id": e.telegram_id} for e in rows]

@app.post("/api/employees")
async def add_employee(data: dict, db=Depends(get_db)):
    e = Employee(
        name        = data.get("name", ""),
        department  = data.get("department", ""),
        position    = data.get("position", ""),
        phone       = data.get("phone", ""),
        email       = data.get("email", ""),
        telegram_id = data.get("telegram_id", ""),
    )
    db.add(e); db.commit(); db.refresh(e)
    return {"id": e.id, "status": "created"}

@app.put("/api/employees/{eid}")
async def update_employee(eid: int, data: dict, db=Depends(get_db)):
    e = db.query(Employee).filter(Employee.id == eid).first()
    if not e:
        raise HTTPException(404, "Not found")
    for field in ["name", "department", "position", "phone", "email", "telegram_id"]:
        if field in data:
            setattr(e, field, data[field])
    db.commit()
    return {"status": "updated"}

@app.delete("/api/employees/{eid}")
async def delete_employee(eid: int, db=Depends(get_db)):
    e = db.query(Employee).filter(Employee.id == eid).first()
    if e:
        e.is_active = False
        db.commit()
    return {"status": "deleted"}

@app.post("/api/employees/{eid}/face")
async def upload_emp_face(eid: int, file: UploadFile = File(...), db=Depends(get_db)):
    import cv2, numpy as np
    data  = await file.read()
    arr   = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Ảnh không hợp lệ")
    ok = face.save_face(db, eid, "employee", frame)
    if not ok:
        raise HTTPException(400, "Không tìm thấy khuôn mặt trong ảnh")
    return {"status": "saved"}

# Guests
@app.get("/api/guests")
async def list_guests(db=Depends(get_db)):
    rows = db.query(Guest).order_by(Guest.last_visit.desc()).limit(50).all()
    return [{"id": g.id, "name": g.name, "company": g.company,
             "visits": g.visit_count, "last": str(g.last_visit)} for g in rows]

# Knowledge Base
@app.post("/api/knowledge/upload")
async def upload_doc(file: UploadFile = File(...), db=Depends(get_db)):
    safe_name = file.filename.replace(" ", "_")
    dest      = Path("data/knowledge") / safe_name
    dest.write_bytes(await file.read())

    doc = KnowledgeDoc(
        filename  = file.filename,
        file_path = str(dest),
        file_type = dest.suffix[1:].lower(),
        status    = "processing")
    db.add(doc); db.commit()

    asyncio.create_task(_ingest(doc.id, str(dest)))
    return {"status": "processing", "doc_id": doc.id}

async def _ingest(doc_id: int, path: str):
    db  = SessionLocal()
    doc = db.query(KnowledgeDoc).filter(KnowledgeDoc.id == doc_id).first()
    if not doc:
        db.close(); return
    try:
        n           = await kb.ingest_file(path, f"doc_{doc_id}")
        doc.status  = "ready" if n > 0 else "error"
        doc.chunk_count   = max(1, n)
        doc.processed_at  = datetime.utcnow()
        db.commit()
        await _broadcast_admin({"type": "doc_ready", "doc_id": doc_id, "status": doc.status})
    except Exception as e:
        doc.status = "error"; db.commit()
        log.error(f"Ingest error: {e}")
    finally:
        db.close()

@app.get("/api/knowledge/docs")
async def list_docs(db=Depends(get_db)):
    rows = db.query(KnowledgeDoc).order_by(KnowledgeDoc.uploaded_at.desc()).all()
    return [{"id": d.id, "filename": d.filename, "type": d.file_type,
             "status": d.status, "chunks": d.chunk_count,
             "uploaded": str(d.uploaded_at)} for d in rows]

@app.delete("/api/knowledge/docs/{did}")
async def delete_doc(did: int, db=Depends(get_db)):
    doc = db.query(KnowledgeDoc).filter(KnowledgeDoc.id == did).first()
    if doc:
        Path(doc.file_path).unlink(missing_ok=True)
        db.delete(doc); db.commit()
    return {"status": "deleted"}

# Appointments
@app.get("/api/appointments")
async def list_appointments(db=Depends(get_db)):
    rows = db.query(Appointment).order_by(Appointment.scheduled_at.desc()).limit(30).all()
    return [{"id": a.id, "guest": a.guest_name, "employee": a.employee_name,
             "time": str(a.scheduled_at), "status": a.status,
             "purpose": a.purpose} for a in rows]

# Telegram
@app.post("/api/telegram/webhook")
async def tg_webhook(update: dict):
    if telegram:
        await telegram.handle_update(update)
    return {"ok": True}

@app.get("/api/telegram/status")
async def tg_status():
    if not telegram or not telegram.token:
        return {"status": "no_token",
                "message": "Chưa cài TELEGRAM_BOT_TOKEN trong .env"}
    return await telegram.get_info()

@app.post("/api/telegram/test")
async def tg_test(data: dict):
    chat_id = data.get("chat_id", "")
    msg     = data.get("message", "Test từ Aria 🤖")
    if not telegram or not telegram.token:
        raise HTTPException(503, "Telegram chưa cài token")
    ok = await telegram.send(chat_id, msg)
    return {"status": "sent" if ok else "failed"}

# Voice
@app.post("/api/voice/settings")
async def voice_settings(data: dict):
    if tts:
        tts.update(
            speed = data.get("speed"),
            lang  = data.get("lang"))
    return {"status": "updated"}

@app.get("/api/voice/preview")
async def voice_preview(text: str = "Xin chào, em là Aria!"):
    if not tts:
        raise HTTPException(503, "TTS not ready")
    audio = await tts.speak(text)
    if not audio:
        raise HTTPException(500, "TTS synthesis failed")
    return Response(content=audio, media_type="audio/mpeg")

# Admin say
@app.post("/api/admin/say")
async def admin_say(data: dict):
    text = data.get("text", "").strip()
    if not text:
        raise HTTPException(400, "text required")
    await _broadcast_tablet({"type": "admin_say", "text": text, "speak": True})
    audio = await tts.speak(text)
    if audio:
        await _broadcast_tablet(
            {"type": "audio", "data": base64.b64encode(audio).decode()})
    return {"status": "ok"}

# ── Frontend ─────────────────────────────────────────────────────────
def _html(rel: str) -> str:
    p = Path(__file__).parent.parent / rel
    return p.read_text(encoding="utf-8") if p.exists() else f"<h2>File not found: {rel}</h2>"

@app.get("/",       response_class=HTMLResponse)
async def root():    return HTMLResponse(_html("frontend-tablet/index.html"))

@app.get("/tablet", response_class=HTMLResponse)
async def tablet():  return HTMLResponse(_html("frontend-tablet/index.html"))

@app.get("/catalog", response_class=HTMLResponse)
async def catalog(): return HTMLResponse(_html("frontend-tablet/catalog.html"))

@app.get("/admin",  response_class=HTMLResponse)
async def admin():   return HTMLResponse(_html("frontend-admin/index.html"))
