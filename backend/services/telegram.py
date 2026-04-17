"""Telegram Bot — thông báo nhân viên, 2 chiều reply"""
import os, json, asyncio, logging
from typing import Optional, Callable, Dict
import httpx

log = logging.getLogger("telegram")


class Telegram:
    def __init__(self):
        self.token    = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.admin_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
        self.on_reply: Optional[Callable] = None   # callback(text, emp_name)
        self.broadcast: Optional[Callable] = None  # broadcast to tablet
        self._replies: Dict[str, str] = {}         # msg_id → emp_name
        self._offset  = 0
        self._running = False
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self):
        if not self.token:
            log.info("Telegram: no token, skipped")
            return
        self._client = httpx.AsyncClient(timeout=30.0)
        me = await self._call("getMe")
        if me.get("ok"):
            log.info(f"Telegram @{me['result']['username']} ready")
            webhook = os.getenv("TELEGRAM_WEBHOOK_URL", "")
            if webhook:
                await self._call("setWebhook", {"url": f"{webhook}/api/telegram/webhook"})
            else:
                asyncio.create_task(self._poll())

    async def stop(self):
        self._running = False
        if self._client:
            await self._client.aclose()

    # ── Gửi thông báo ──────────────────────────────────────────────
    async def notify(self, chat_id: str, emp_name: str,
                     guest_name: str, purpose: str = "") -> Optional[str]:
        from datetime import datetime
        text = (f"🔔 *Có khách chờ gặp {emp_name}*\n\n"
                f"👤 Khách: *{guest_name or 'Khách lạ'}*\n"
                f"💬 {purpose[:80] if purpose else 'Không rõ'}\n"
                f"⏰ {datetime.now().strftime('%H:%M %d/%m/%Y')}\n\n"
                f"_Reply tin nhắn này để Aria nói lại với khách_")
        kb = {"inline_keyboard": [[
            {"text": "🚶 Tôi đang đến", "callback_data": "coming"},
            {"text": "⏳ Nhờ chờ 5p",  "callback_data": "wait5"},
        ]]}
        r = await self._call("sendMessage", {
            "chat_id": chat_id, "text": text,
            "parse_mode": "Markdown",
            "reply_markup": json.dumps(kb)
        })
        if r.get("ok"):
            msg_id = str(r["result"]["message_id"])
            self._replies[msg_id] = emp_name
            asyncio.create_task(self._expire_reply(msg_id))
            return msg_id
        return None

    async def _expire_reply(self, msg_id: str):
        await asyncio.sleep(3600)
        self._replies.pop(msg_id, None)

    # ── Webhook / Polling ──────────────────────────────────────────
    async def handle_update(self, update: dict):
        msg = update.get("message", {})
        if not msg:
            # Callback query (button press)
            if "callback_query" in update:
                q = update["callback_query"]
                data  = q.get("data", "")
                cid   = str(q["message"]["chat"]["id"])
                reply = {"coming": "✅ Nhân viên đang đến.",
                         "wait5":  "⏳ Nhân viên nhờ chờ khoảng 5 phút."}
                if data in reply and self.broadcast:
                    await self.broadcast({"type": "admin_say",
                                         "text": reply[data], "speak": True})
                await self._call("answerCallbackQuery",
                                 {"callback_query_id": q["id"], "text": "👍"})
            return

        cid  = str(msg["chat"]["id"])
        text = msg.get("text", "")
        user = msg.get("from", {})
        rep  = msg.get("reply_to_message", {})

        if text.startswith("/"):
            await self._cmd(cid, text, user)
            return

        # Reply từ nhân viên → Aria nói với khách
        if rep and self.on_reply:
            orig = str(rep.get("message_id", ""))
            if orig in self._replies:
                emp_name = self._replies[orig]
                await self.on_reply(text, emp_name)
                await self.send(cid, f'✅ Đã gửi đến khách: "{text}"')

    async def _cmd(self, cid: str, text: str, user: dict):
        cmd = text.split()[0].lower().replace("/", "").split("@")[0]
        if cmd in ("start", "help"):
            await self.send(cid,
                "🤖 *Aria Lễ Tân*\n\n"
                "/setid — Lấy Chat ID\n"
                "/status — Trạng thái\n"
                "/say [text] — Phát qua loa\n\n"
                "_Reply thông báo có khách để gửi phản hồi_", md=True)
        elif cmd == "setid":
            await self.send(cid,
                f"Chat ID: `{cid}`\n"
                f"Tên: {user.get('first_name','')} {user.get('last_name','')}\n"
                f"Copy số này → Admin → Nhân viên → Telegram ID", md=True)
        elif cmd == "status":
            from datetime import datetime
            await self.send(cid,
                f"🟢 Aria đang hoạt động\n⏰ {datetime.now().strftime('%H:%M %d/%m')}")
        elif cmd == "say":
            parts = text.split(maxsplit=1)
            if len(parts) > 1 and self.broadcast:
                await self.broadcast({"type": "admin_say", "text": parts[1], "speak": True})
                await self.send(cid, f"🔊 Đang phát: {parts[1]}")

    async def send(self, chat_id: str, text: str, md: bool = False) -> bool:
        r = await self._call("sendMessage", {
            "chat_id": chat_id, "text": text,
            **({"parse_mode": "Markdown"} if md else {})
        })
        return r.get("ok", False)

    async def get_info(self) -> dict:
        r = await self._call("getMe")
        if r.get("ok"):
            b = r["result"]
            return {"status": "ok", "username": b["username"], "name": b["first_name"]}
        return {"status": "error", "message": "Token không hợp lệ"}

    async def _poll(self):
        self._running = True
        while self._running:
            try:
                r = await self._call("getUpdates", {
                    "offset": self._offset + 1, "timeout": 20,
                    "allowed_updates": ["message", "callback_query"]
                })
                for upd in r.get("result", []):
                    self._offset = upd["update_id"]
                    await self.handle_update(upd)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Telegram poll error: {e}")
                await asyncio.sleep(5)

    async def _call(self, method: str, params: dict = None) -> dict:
        if not self._client:
            self._client = httpx.AsyncClient(timeout=30.0)
        try:
            r = await self._client.post(
                f"https://api.telegram.org/bot{self.token}/{method}",
                json=params or {})
            return r.json()
        except Exception as e:
            log.error(f"Telegram {method}: {e}")
            return {"ok": False}
