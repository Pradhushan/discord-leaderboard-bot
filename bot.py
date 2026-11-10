import os, io, sys, json, time, base64, asyncio, random, traceback, html as htmllib
import datetime as dt
from zoneinfo import ZoneInfo

import aiohttp
import discord
from discord import app_commands
from discord.utils import escape_markdown
from aiohttp import web
from google import genai
from google.genai import types
from playwright.async_api import async_playwright

# ---------- CONFIG ----------
TZ = ZoneInfo("Asia/Colombo")
TOKEN = os.environ["DISCORD_TOKEN"]
GUILD_ID = int(os.environ["GUILD_ID"])
PROGRESS_CH = int(os.environ["PROGRESS_CHANNEL_ID"])  # progress + leaderboard live here
DATA_CH = int(os.environ["DATA_CHANNEL_ID"])          # private channel where the bot saves its data
MEDIA_CH = int(os.environ.get("MEDIA_CHANNEL_ID", "0"))  # "aah pics" (media only)
STUDENT_ROLE = int(os.environ["STUDENT_ROLE_ID"])
ROLE_MODEL_ID = int(os.environ["ROLE_MODEL_ID"])      # NotYourFriend / Tharuka
ROLE_MODEL_MIN = 600
ROLE_MODEL_CAPTION = "Catch me if u can! 🏎️💨"
BASELINE = int(os.environ.get("BASELINE_MINUTES", "240"))
EXAM_DATE = dt.date.fromisoformat(os.environ.get("EXAM_DATE", "2027-08-03"))
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACKS = [m.strip() for m in os.environ.get("GEMINI_FALLBACKS", "").split(",") if m.strip()]
RETRY_DELAYS = [2, 5, 10, 20]
REPOST = os.environ.get("REPOST_ON_UPDATE", "0") == "1"
ai = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
NO_MENTIONS = discord.AllowedMentions.none()

STATE = {}
LOCK = asyncio.Lock()         # protects STATE
RENDER_LOCK = asyncio.Lock()  # only one Chromium at a time
last_data_msg = None
last_tick = 0.0
_post_fails = 0
backoff_until = 0.0           # when Discord blocks us (429), the scheduler waits until this time
POST_PROGRESS = {"img": False, "parts": 0}  # so a retry never reposts what was already sent
BG_TASKS = set()

def spawn(coro):
    t = asyncio.create_task(coro)
    BG_TASKS.add(t)
    t.add_done_callback(BG_TASKS.discard)

def is_429(ex):
    return isinstance(ex, discord.HTTPException) and ex.status == 429

def now():
    return dt.datetime.now(TZ)

def fresh_state(day: dt.date):
    return {"day": day.isoformat(), "msg_id": None, "entries": {},
            "closed": False, "result": None, "posted": False}

def fmt_hm(m): return f"{m // 60}hrs {m % 60}mins"      # progress message
def fmt(m): return f"{m // 60:02d}h {m % 60:02d}m"       # leaderboard
def fmt_short(m): return f"{m // 60}h" if m % 60 == 0 else f"{m // 60}h {m % 60}m"

def ordinal(n):
    suf = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"

def chunk(text: str, limit: int = 1900):
    parts, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit and cur:
            parts.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur)
    return parts

# ---------- AI (captions only) ----------
async def ask_json(prompt: str, temperature: float):
    last_err = None
    for model in [MODEL] + FALLBACKS:
        for attempt in range(len(RETRY_DELAYS) + 1):
            def call(model=model):
                r = ai.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", temperature=temperature),
                )
                return json.loads(r.text)
            try:
                return await asyncio.to_thread(call)
            except Exception as ex:
                last_err = ex
                msg = str(ex)
                retryable = isinstance(ex, (ValueError, TypeError)) or any(
                    k in msg for k in ("503", "429", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
                if not retryable or attempt == len(RETRY_DELAYS):
                    break
                await asyncio.sleep(RETRY_DELAYS[attempt])
    raise last_err

CAPTION_PROMPT = """You write one-line captions for a Sri Lankan A/L study-group leaderboard.
You are a funny friend in the group chat, hyping or roasting each person based on THEIR day.

Voice:
- Real Sri Lankan teen Singlish, the way friends text. Sinhala script mixed with English study words
  (block, set, past paper, baseline, afk, grind, chapter names).
- Casual slang, e.g. machan, goda, ගැහුවා, අතුගාලා, ඉවරයි, පට්ට, ඔහොමම යං.
  These are only vocabulary hints. Use whatever fits naturally and don't force them in.
- NEVER formal or textbook Sinhala. Never sound like a translation or a motivational poster.

Data fields: "title" is empty.
"topics" is the work they logged, each with its time.

Content rules:
- Every caption must reference something specific from that person's data: a topic or the hours. No generic lines that could fit anyone.
- passed=true: hype and respect. passed=false: light, friendly roast.
- If minutes is 0 and topics is ["no post"], they logged nothing. Roast the silence.
- Max ~12 words. End with exactly two emojis. No hashtags, no quotes, no asterisks or underscores.
- Never reuse wording across people in the same list. Every caption must be different.

Return ONLY a JSON list: [{"id": "...", "caption": "..."}]
Data (baseline_minutes is the pass mark):
"""

async def write_captions(rows):
    data = [{"id": str(r["uid"]), "name": r["name"], "minutes": r["minutes"], "passed": r["passed"],
             "title": r["title"], "topics": r["topics"], "baseline_minutes": BASELINE}
            for r in rows if r["uid"] != ROLE_MODEL_ID]
    if not data:
        return {}
    try:
        out = await ask_json(CAPTION_PROMPT + json.dumps(data, ensure_ascii=False), 1.0)
        if isinstance(out, dict):
            out = [out]
        return {int(o["id"]): o["caption"].strip() for o in out}
    except Exception:
        traceback.print_exc()
        return {}

def clean_caption(c: str) -> str:
    return " ".join(c.replace("*", "").replace("_", " ").replace("\n", " ").split())

# ---------- STORAGE (kept in a private Discord channel so restarts never lose data) ----------
async def get_ch(cid):
    return bot.get_channel(cid) or await bot.fetch_channel(cid)

async def save_state():
    global last_data_msg
    ch = await get_ch(DATA_CH)
    blob = json.dumps(STATE, ensure_ascii=False).encode("utf-8")
    new = await ch.send("💾 bot data (please don't delete)",
                        file=discord.File(io.BytesIO(blob), "state.json"))
    old, last_data_msg = last_data_msg, new
    if old:
        try:
            await old.delete()
        except discord.HTTPException:
            pass

async def load_state():
    global last_data_msg
    ch = await get_ch(DATA_CH)
    async for m in ch.history(limit=50):  # newest first
        if m.author.id == bot.user.id:
            att = next((a for a in m.attachments if a.filename == "state.json"), None)
            if att:
                try:
                    data = json.loads((await att.read()).decode("utf-8"))
                    STATE.clear()
                    STATE.update(data)
                    last_data_msg = m
                    break
                except Exception:
                    traceback.print_exc()
    if not STATE:
        STATE.update(fresh_state(now().date()))
        await save_state()
    for k, v in fresh_state(now().date()).items():
        STATE.setdefault(k, v)

# ---------- PROGRESS MESSAGE ----------
def students(guild):
    role = guild.get_role(STUDENT_ROLE)
    if not role:
        return []
    return [m for m in role.members if m.id != ROLE_MODEL_ID and not m.bot]

def render_progress(guild):
    custom_order = ["★pedro★⋆˚꩜｡", "schrödinger's cat", "Dr. Novice"]
    members = sorted(
        students(guild),
        key=lambda m: custom_order.index(m.display_name) if m.display_name in custom_order else 999
    )

    day = dt.date.fromisoformat(STATE["day"])
    header = f"✨ Progress\n📅 Date: {ordinal(day.day)} {day.strftime('%b %Y')}\n━━━━━━━━━━━━━━━━━━━\n\n"

    def build(keep):
        blocks = []
        for mem in members:
            uid = str(mem.id)
            items = STATE["entries"].get(uid, [])
            total = sum(i[1] for i in items)
            name = mem.display_name.replace("`", "'")
            lines = [f"📌 **{name}** • *{fmt_hm(total)}*"]
            start = max(0, len(items) - keep) if keep else 0
            if start:
                lines.append(f"\n            … {start} earlier")
            for i in range(start, len(items)):
                task, mins = items[i]
                lines.append(f"\n            {i + 1}. {escape_markdown(task)} — `{fmt_hm(mins)}`")
            blocks.append("".join(lines))
        return header + ("\n\n".join(blocks) or "No students found.")

    text = build(None)
    for keep in (20, 12, 8, 5, 3, 1):  # shrink if it would exceed Discord's 2000-char limit
        if len(text) <= 1990:
            break
        text = build(keep)
    return text[:1990]

async def refresh_progress(guild):
    text = render_progress(guild)
    ch = await get_ch(PROGRESS_CH)
    msg_id = STATE.get("msg_id")
    done = False
    if msg_id:
        part = ch.get_partial_message(msg_id)
        try:
            if REPOST:
                await part.delete()
            else:
                await part.edit(content=text, allowed_mentions=NO_MENTIONS)
                done = True
        except discord.NotFound:
            pass
    if not done:
        new = await ch.send(text, allowed_mentions=NO_MENTIONS)
        STATE["msg_id"] = new.id
        await save_state()

# ---------- COMMAND HELPERS ----------
async def _del_later(inter, secs):
    await asyncio.sleep(secs)
    try:
        await inter.delete_original_response()
    except Exception:
        pass

async def tell(inter, text, secs=60):
    """Private (ephemeral) reply that deletes itself after `secs` seconds."""
    await inter.followup.send(text, ephemeral=True)
    spawn(_del_later(inter, secs))

class guarded:
    def __init__(self, inter): self.inter = inter
    async def __aenter__(self): return self
    async def __aexit__(self, et, ev, tb):
        if isinstance(ev, Exception):
            traceback.print_exception(et, ev, tb)
            try:
                await tell(self.inter, f"⚠️ Something went wrong: `{ev}`", 30)
            except Exception:
                pass
            return True
        return False

async def allowed(inter) -> bool:
    u = inter.user
    if u.id == ROLE_MODEL_ID:
        await tell(inter, "😎 You're locked in at 10h. No need to log anything.", 15)
        return False
    if not (isinstance(u, discord.Member) and u.get_role(STUDENT_ROLE)):
        await tell(inter, "⚠️ Only members with the Student role can use this.", 15)
        return False
    return True

def day_open():
    return (not STATE["closed"]) and STATE["day"] == now().date().isoformat()

CLOSED_MSG = "🛑 Pens down! Today's board is closed. You can log again after the leaderboard is posted."

# ---------- BOT ----------
intents = discord.Intents.default()
intents.members = True
intents.message_content = True

class Bot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        if os.environ.get("SYNC_COMMANDS", "1") == "1":  # set to 0 after commands appear
            g = discord.Object(GUILD_ID)
            self.tree.copy_global_to(guild=g)
            await self.tree.sync(guild=g)
        await load_state()
        app = web.Application()
        app.router.add_get("/", lambda r: web.Response(text="ok"))
        app.router.add_get("/health", health)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", int(os.environ.get("PORT", 10000))).start()
        spawn(scheduler())
        spawn(watchdog())
        spawn(keepalive())

    async def on_message(self, m: discord.Message):
        if m.author.bot or not MEDIA_CH or m.channel.id != MEDIA_CH:
            return
        if m.type not in (discord.MessageType.default, discord.MessageType.reply):
            return
        try:
            await handle_media_channel(m)
        except Exception:
            traceback.print_exc()

bot = Bot()

# ---------- COMMANDS ----------
@bot.tree.command(name="progress", description="Log what you studied")
@app_commands.describe(task="What you studied", hours="Hours spent", minutes="Minutes spent")
@app_commands.guild_only()
async def progress_cmd(inter: discord.Interaction, task: app_commands.Range[str, 1, 50],
                       hours: app_commands.Range[int, 0, 24], minutes: app_commands.Range[int, 0, 59]):
    await inter.response.defer(ephemeral=True)
    async with guarded(inter):
        if not await allowed(inter):
            return
        uid = str(inter.user.id)
        mins = hours * 60 + minutes
        async with LOCK:
            if not day_open():
                await tell(inter, CLOSED_MSG, 20)
                return
            if mins <= 0:
                await tell(inter, "⚠️ Time must be more than 0 minutes.", 15)
                return
            items = STATE["entries"].setdefault(uid, [])
            if sum(i[1] for i in items) + mins > 1440:
                await tell(inter, "⚠️ That would add up to more than 24 hours today.", 20)
                return
            items.append([task.strip(), mins])
            n = len(items)
            await save_state()
            await refresh_progress(inter.guild)
        await tell(inter, f"✅ Added #{n}: {task} ({fmt_hm(mins)})", 10)

@bot.tree.command(name="delete", description="Remove one of your progress entries")
@app_commands.describe(number="The number shown next to the entry in the progress message")
@app_commands.guild_only()
async def delete_cmd(inter: discord.Interaction, number: app_commands.Range[int, 1, 200]):
    await inter.response.defer(ephemeral=True)
    async with guarded(inter):
        if not await allowed(inter):
            return
        uid = str(inter.user.id)
        async with LOCK:
            if not day_open():
                await tell(inter, CLOSED_MSG, 20)
                return
            items = STATE["entries"].get(uid, [])
            if number > len(items):
                await tell(inter, f"⚠️ You don't have an entry #{number}.", 15)
                return
            task, _ = items.pop(number - 1)
            await save_state()
            await refresh_progress(inter.guild)
        await tell(inter, f"🗑️ Removed #{number}: {task}", 10)

@bot.tree.command(name="testnow", description="Post a test leaderboard from today's progress")
@app_commands.default_permissions(administrator=True)
@app_commands.guild_only()
async def testnow(inter: discord.Interaction):
    await inter.response.defer(ephemeral=True)
    async with guarded(inter):
        if RENDER_LOCK.locked():
            await tell(inter, "⏳ A leaderboard is already being generated. Try again shortly.", 15)
            return
        async with RENDER_LOCK:
            result = await compute_result(inter.guild, now().date())
            await post_result(inter.guild, result, prefix="🧪 **TEST RUN**")
        await tell(inter, f"✅ Posted in <#{PROGRESS_CH}>", 15)

# ---------- LEADERBOARD ----------
async def compute_result(guild, day: dt.date):
    role = guild.get_role(STUDENT_ROLE)
    members = {m.id: m for m in role.members} if role else {}
    if ROLE_MODEL_ID not in members:
        members[ROLE_MODEL_ID] = guild.get_member(ROLE_MODEL_ID) or await guild.fetch_member(ROLE_MODEL_ID)

    rows = []
    for uid, mem in members.items():
        if uid == ROLE_MODEL_ID:
            mins, title, topics = ROLE_MODEL_MIN, "", []
        else:
            items = STATE["entries"].get(str(uid), [])
            mins = min(sum(i[1] for i in items), 1440)
            title = ""
            topics = [f"{t} ({fmt_short(m)})" for t, m in items] or ["no post"]
        rows.append({"uid": uid, "name": mem.display_name, "minutes": mins, "passed": mins >= BASELINE,
                     "title": title, "topics": topics,
                     "avatar": str(mem.display_avatar.replace(size=128, format="png").url)})
    rows.sort(key=lambda r: -r["minutes"])

    caps = await write_captions(rows)
    for r in rows:
        r["caption"] = ROLE_MODEL_CAPTION if r["uid"] == ROLE_MODEL_ID \
            else clean_caption(caps.get(r["uid"], "📝⚡"))
    return {"day": day.isoformat(), "days_left": max(0, (EXAM_DATE - day).days), "rows": rows}

def result_text(result):
    day = dt.date.fromisoformat(result["day"])
    rows = result["rows"]
    passed = [r for r in rows if r["passed"]]
    failed = [r for r in rows if not r["passed"]]

    def line(r):
        mark = "✅" if r["passed"] else "❌"
        return f"{mark} **{escape_markdown(r['name'])}** (**{fmt(r['minutes'])}**): *{r['caption']}*"

    lines = [f"🔥 _*THE GRIND*_ 🔥 — {ordinal(day.day)} {day.strftime('%b')}",
             f"⏳ {result['days_left']} DAYS MORE!", ""]
    lines += [line(r) for r in passed]
    if passed and failed:
        lines.append("-")
    lines += [line(r) for r in failed]
    lines += ["", f"_*{fmt_short(BASELINE)} baseline. {len(passed)} passed, {len(failed)} failed.*_"]
    return "\n".join(lines)

BG_SVG = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='60' height='60' viewBox='0 0 60 60'><path d='M-15,45 L45,-15 M0,60 L60,0 M15,75 L75,15' stroke='%23ff2a2a' stroke-width='1.2' opacity='0.12'/><circle cx='30' cy='30' r='1.5' fill='%23ff2a2a' opacity='0.3'/></svg>"
PLACEHOLDER = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='76' height='76'><rect width='76' height='76' fill='%23444'/></svg>"
RANK_C = ["#ffca28", "#e0e0e0", "#cd7f32", "#ff5252", "#ff5252"]
NAME_C = ["#ffca28", "#ffffff", "#cd7f32", "#ff5252", "#ff5252"]

HEAD = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Anonymous+Pro:wght@400;700&display=swap');
body{margin:0;padding:0;background:#121212}
#capture-area{width:1080px;height:1080px;background-color:#141414;
background-image:radial-gradient(circle at 50% 45%,rgba(255,42,42,0.09) 0%,rgba(18,18,18,0.95) 70%),url("__BG__");
background-repeat:no-repeat,repeat;color:white;font-family:'Anonymous Pro',monospace;box-sizing:border-box;
display:flex;flex-direction:column;align-items:center;justify-content:center;gap:50px;padding:40px 0;position:relative}
.header{text-align:center;display:flex;flex-direction:column;align-items:center}
.title-main{font-family:'Orbitron',sans-serif;color:#ff2a2a;font-size:58px;letter-spacing:8px;margin:0;font-weight:900;
text-shadow:0 0 25px rgba(255,42,42,0.75),0 0 10px rgba(255,42,42,0.5)}
.title-sub{font-family:'Orbitron',sans-serif;color:#ffca28;font-size:36px;letter-spacing:12px;margin:4px 0 0 0;font-weight:700;
text-shadow:0 0 15px rgba(255,202,40,0.5)}
.leaderboard-container{display:flex;flex-direction:column;gap:20px;width:820px}
.card{background:rgba(26,26,26,0.95);border-radius:35px;padding:0 45px;height:110px;display:flex;align-items:center;
border:1px solid rgba(255,42,42,0.35);box-shadow:0 0 25px 4px rgba(255,42,42,0.18);box-sizing:border-box}
.rank{font-size:36px;font-weight:700;width:90px;letter-spacing:1px}
.profile-pic{width:76px;height:76px;border-radius:50%;object-fit:cover;margin-right:35px;border:none}
.name{font-size:32px;font-weight:700;letter-spacing:3px;flex-grow:1;overflow:hidden;white-space:nowrap}
.time{font-size:32px;font-weight:700;color:#ffca28;letter-spacing:2px}
.empty{font-size:30px;font-weight:700;letter-spacing:3px;color:#ff5252;justify-content:center}
.footer{font-family:'Orbitron',sans-serif;font-size:24px;letter-spacing:8px;color:#fff;font-weight:700;
text-shadow:0 0 10px rgba(255,255,255,0.5)}
</style></head><body><div id="capture-area">
<div class="header"><h1 class="title-main">THE GRIND</h1><h2 class="title-sub">LEADERBOARD</h2></div>
<div class="leaderboard-container">""".replace("__BG__", BG_SVG)

BROWSER_ARGS = ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
                "--disable-extensions", "--disable-background-networking"]

async def avatar_uri(session, url):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
            data = await r.read()
            ctype = r.headers.get("Content-Type", "image/png")
        return f"data:{ctype};base64," + base64.b64encode(data).decode()
    except Exception:
        return PLACEHOLDER

async def render(passed_rows, days_left: int) -> bytes:
    cards = ""
    async with aiohttp.ClientSession() as session:
        for i, r in enumerate(passed_rows[:5]):  # image: only baseline-clearers, max 5
            av = await avatar_uri(session, r["avatar"])
            h, m = divmod(r["minutes"], 60)
            cards += (f'<div class="card"><div class="rank" style="color:{RANK_C[i]}">#{i+1}</div>'
                      f'<img class="profile-pic" src="{av}" alt="">'
                      f'<div class="name" style="color:{NAME_C[i]}">{htmllib.escape(r["name"].upper())}</div>'
                      f'<div class="time">{h:02d}H {m:02d}M</div></div>')
    if not cards:
        cards = '<div class="card empty">NOBODY CLEARED THE BASELINE 💀</div>'
    page = HEAD + cards + f'</div><div class="footer">{days_left} DAYS MORE!</div></div></body></html>'
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=BROWSER_ARGS)
        try:
            pg = await browser.new_page(viewport={"width": 1080, "height": 1080})
            await pg.set_content(page, wait_until="load", timeout=30000)
            try:
                await asyncio.wait_for(pg.evaluate("document.fonts.ready"), timeout=10)
            except Exception:
                pass
            png = await pg.locator("#capture-area").screenshot()
        finally:
            await browser.close()
    return png

async def post_result(guild, result, prefix: str = "", track=False):
    """Posts image first, captions after. With track=True, a retry never reposts what already went out."""
    ch = await get_ch(PROGRESS_CH)
    if not (track and POST_PROGRESS["img"]):
        png = None
        try:
            png = await render([r for r in result["rows"] if r["passed"]], result["days_left"])
        except Exception:
            traceback.print_exc()
        if png:
            await ch.send(content=prefix or None, file=discord.File(io.BytesIO(png), "leaderboard.png"))
        else:
            await ch.send((prefix + "\n" if prefix else "") + "⚠️ Couldn't generate the leaderboard image this time.")
        if track:
            POST_PROGRESS["img"] = True
    parts = chunk(result_text(result))
    start = POST_PROGRESS["parts"] if track else 0
    for i in range(start, len(parts)):
        await ch.send(parts[i], allowed_mentions=NO_MENTIONS)
        if track:
            POST_PROGRESS["parts"] = i + 1

# ---------- SCHEDULER (checks every 20s, so a restart never misses the day) ----------
PENS_DOWN = [
    "✏️ Pens down, people! Thanks for the hard work today. The leaderboard will be out in a short while ⏳",
    "🛑 Pens down! Great grind today, everyone. Leaderboard coming up shortly 🔥",
    "📕 Books closed for today! Thank you all for working hard. The leaderboard will be out in a few minutes ⚡",
    "😮‍💨 That's a wrap, team! Proud of you all. The leaderboard drops in a short while 🏁",
]

async def roll_to_today():
    async with LOCK:
        STATE.clear()
        STATE.update(fresh_state(now().date()))
        await save_state()

async def tick():
    global last_tick, _post_fails, backoff_until
    last_tick = time.time()
    if time.time() < backoff_until:  # Discord blocked us recently, wait it out
        return
    guild = bot.get_guild(GUILD_ID)
    if guild is None or not STATE:
        return
    n = now()
    day = dt.date.fromisoformat(STATE["day"])

    if STATE["posted"]:  # crashed between posting and rolling over
        if day < n.date():
            await roll_to_today()
        return

    # 1) close the day at 23:55 (or immediately if we're catching up late)
    if not STATE["closed"] and (day < n.date() or n.time() >= dt.time(23, 55)):
        if day == n.date():
            ch = await get_ch(PROGRESS_CH)
            await ch.send(random.choice(PENS_DOWN), allowed_mentions=NO_MENTIONS)
        async with LOCK:
            STATE["closed"] = True
            await save_state()

    # 2) work out the results (AI captions)
    if STATE["closed"] and STATE["result"] is None:
        result = await compute_result(guild, day)
        async with LOCK:
            STATE["result"] = result
            await save_state()

    # 3) post at 00:01
    if STATE["result"] and not STATE["posted"]:
        post_time = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(0, 1), TZ)
        if n >= post_time:
            try:
                async with RENDER_LOCK:
                    await post_result(guild, STATE["result"], track=True)
            except Exception as ex:
                traceback.print_exc()
                if is_429(ex):
                    backoff_until = time.time() + 600  # blocked: wait 10 min, keep trying (never give up)
                    return
                _post_fails += 1
                backoff_until = time.time() + 60
                if _post_fails < 3:
                    return
                try:
                    ch = await get_ch(PROGRESS_CH)
                    await ch.send(f"⚠️ Leaderboard failed tonight: `{ex}`")
                except Exception:
                    pass
            _post_fails = 0
            POST_PROGRESS.update(img=False, parts=0)
            async with LOCK:
                STATE["posted"] = True
                await save_state()
            await roll_to_today()

async def scheduler():
    global backoff_until
    await bot.wait_until_ready()
    while True:
        try:
            await tick()
        except Exception as ex:
            traceback.print_exc()
            if is_429(ex):
                backoff_until = time.time() + 600
        await asyncio.sleep(20)

# ---------- HEALTH / KEEP-ALIVE ----------
async def health(request):
    ok = bot.is_ready() and not bot.is_closed() and (last_tick == 0 or time.time() - last_tick < 1200)
    return web.Response(text="ok" if ok else "unhealthy", status=200 if ok else 503)

async def watchdog():
    """If Discord stays disconnected (or the scheduler freezes), exit so Render restarts us."""
    bad_since = None
    while True:
        await asyncio.sleep(30)
        if bot.is_ready() and not bot.is_closed():
            bad_since = None
        else:
            bad_since = bad_since or time.time()
            if time.time() - bad_since > 300:
                print("Watchdog: Discord connection dead, restarting.", flush=True)
                os._exit(1)
        if last_tick and time.time() - last_tick > 1200:
            print("Watchdog: scheduler frozen, restarting.", flush=True)
            os._exit(1)

async def keepalive():
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        return
    async with aiohttp.ClientSession() as s:
        while True:
            await asyncio.sleep(600)
            try:
                await s.get(url, timeout=aiohttp.ClientTimeout(total=15))
            except Exception:
                pass

# ---------- MEDIA-ONLY CHANNEL ----------
MEDIA_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic", ".avif",
             ".mp4", ".mov", ".webm", ".mkv", ".m4v")

def has_media(m: discord.Message) -> bool:
    for a in m.attachments:
        ct = (a.content_type or "").lower()
        if ct.startswith(("image/", "video/")) or a.filename.lower().endswith(MEDIA_EXT):
            return True
    return any(e.type in ("image", "video", "gifv") for e in m.embeds)

async def handle_media_channel(m: discord.Message):
    if has_media(m):
        return
    if "http" in m.content.lower():  # link previews (gifs etc.) take a moment to appear
        await asyncio.sleep(3)
        try:
            m = await m.channel.fetch_message(m.id)
        except discord.NotFound:
            return
        except discord.HTTPException:
            pass
        if has_media(m):
            return
    try:
        await m.delete()
    except discord.Forbidden:
        print("Missing 'Manage Messages' permission in the media channel.", flush=True)
        return
    except discord.NotFound:
        pass
    warn = (f"📸 {m.channel.mention} only allows photos or videos (a caption with them is fine). "
            "Text-only messages aren't allowed, so I removed yours. This message disappears in a minute.")
    try:
        await m.author.send(warn, delete_after=60)
    except (discord.Forbidden, discord.HTTPException):
        try:  # DMs closed: short public notice instead
            await m.channel.send(f"{m.author.mention} only photos or videos here (captions are fine) 📸",
                                 delete_after=15, allowed_mentions=discord.AllowedMentions(users=[m.author]))
        except discord.HTTPException:
            pass

# ---------- START ----------
try:
    bot.run(TOKEN)
except discord.HTTPException as ex:
    print(f"Discord login failed ({ex.status}).", flush=True)
    time.sleep(900 if ex.status == 429 else 30)  # space out restarts so we don't hammer Discord
    sys.exit(1)