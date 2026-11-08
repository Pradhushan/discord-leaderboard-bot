import os, io, json, base64, asyncio, traceback, html as htmllib
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
PROGRESS_CH = int(os.environ["PROGRESS_CHANNEL_ID"])
BOARD_CH = int(os.environ["LEADERBOARD_CHANNEL_ID"])
STUDENT_ROLE = int(os.environ["STUDENT_ROLE_ID"])
ROLE_MODEL_ID = int(os.environ["ROLE_MODEL_ID"])  # NotYourFriend / Tharuka
ROLE_MODEL_MIN = 600  # his 10h default
ROLE_MODEL_CAPTION = "Catch me if u can! 🏎️💨"  # always this, nothing else
BASELINE = int(os.environ.get("BASELINE_MINUTES", "240"))
EXAM_DATE = dt.date.fromisoformat(os.environ.get("EXAM_DATE", "2027-08-03"))
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACKS = [m.strip() for m in os.environ.get("GEMINI_FALLBACKS", "").split(",") if m.strip()]
RETRY_DELAYS = [2, 5, 10, 20]  # seconds between retries
ai = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
RUN_LOCK = asyncio.Lock()  # only one leaderboard build at a time (saves memory)

# ---------- AI ----------
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
                    break  # give up on this model, try the next one
                await asyncio.sleep(RETRY_DELAYS[attempt])
    raise last_err

EXTRACT_PROMPT = """You read daily study-progress messages posted by Sri Lankan A/L students.
For each message return the TOTAL study time actually completed, plus what they worked on.
Rules:
- If the message states a total (e.g. "Σt = 6hrs 41mins", "Total - 1 hr 25 mins"), use it.
- Otherwise add up the completed items. Skip items marked ❌ or clearly not done.
- "title" is the header/goal line if there is one (e.g. "8hrs+ today? 03.11"), else "".
  The title is NOT study time.
- "topics" is a list of the subjects/chapters studied, each max 4 words, with their minutes if given.
- Text may use fancy unicode fonts (𝓱𝓻 𝓶𝓲𝓷𝓼). Read them normally.
- Units: hr/hrs/h = hours, min/mins/m = minutes.
- confident=false if you had to guess or there is no usable time.
Return ONLY a JSON list:
[{"id": "...", "total_minutes": int, "title": "...", "topics": ["..."], "confident": bool}]
Messages:
"""

CAPTION_PROMPT = """You write one-line captions for a Sri Lankan A/L study-group leaderboard.
You are a funny friend in the group chat, hyping or roasting each person based on THEIR day.

Voice:
- Real Sri Lankan teen Singlish, the way friends text. Sinhala script mixed with English study words
  (block, set, past paper, baseline, afk, grind, chapter names).
- Casual slang, e.g. machan, goda, ගැහුවා, අතුගාලා, ඉවරයි, පට්ට, ඔහොමම යං.
  These are only vocabulary hints. Use whatever fits naturally and don't force them in.
- NEVER formal or textbook Sinhala. Never sound like a translation or a motivational poster.

Content rules:
- Every caption must reference something specific from that person's data: a topic, the hours,
  or their title/goal. No generic lines that could fit anyone.
- passed=true: hype and respect. passed=false: light, friendly roast (if they posted a goal title
  they missed, roast the gap between goal and reality).
- If minutes is 0 and topics is ["no post"], they posted nothing. Roast the silence.
- Max ~12 words. End with exactly two emojis. No hashtags, no quotes, no asterisks or underscores.
- Never reuse wording across people in the same list. Every caption must be different.

Return ONLY a JSON list: [{"id": "...", "caption": "..."}]
Data (baseline_minutes is the pass mark):
"""

async def extract(texts: dict):
    if not texts:
        return {}
    payload = [{"id": str(k), "text": v} for k, v in texts.items()]
    out = await ask_json(EXTRACT_PROMPT + json.dumps(payload, ensure_ascii=False), 0.0)
    if isinstance(out, dict):
        out = [out]
    return {int(o["id"]): o for o in out}

async def write_captions(entries):
    # Tharuka's caption is fixed, so he is not sent to the AI
    rows = [{"id": str(e["uid"]), "name": e["name"], "minutes": e["minutes"],
             "passed": e["passed"], "title": e["title"], "topics": e["topics"],
             "baseline_minutes": BASELINE}
            for e in entries if e["uid"] != ROLE_MODEL_ID]
    if not rows:
        return {}
    try:
        out = await ask_json(CAPTION_PROMPT + json.dumps(rows, ensure_ascii=False), 1.0)
        if isinstance(out, dict):
            out = [out]
        return {int(o["id"]): o["caption"].strip() for o in out}
    except Exception:
        traceback.print_exc()
        return {}

def clean_caption(c: str) -> str:
    # remove characters that would break Discord's italic formatting
    return " ".join(c.replace("*", "").replace("\n", " ").split())

# ---------- COLLECT ----------
async def collect(guild: discord.Guild, day: dt.date, end: dt.datetime):
    start = dt.datetime.combine(day, dt.time.min, TZ)
    ch = guild.get_channel(PROGRESS_CH) or await guild.fetch_channel(PROGRESS_CH)
    last = {}
    async for m in ch.history(after=start, before=end, limit=None, oldest_first=True):
        if not m.author.bot and m.content.strip():
            last[m.author.id] = m.content  # later message overwrites earlier

    role = guild.get_role(STUDENT_ROLE)
    members = {m.id: m for m in role.members}
    if ROLE_MODEL_ID not in members:
        members[ROLE_MODEL_ID] = guild.get_member(ROLE_MODEL_ID) or await guild.fetch_member(ROLE_MODEL_ID)

    texts = {u: t for u, t in last.items() if u in members and u != ROLE_MODEL_ID}
    ext = await extract(texts)

    entries = []
    for uid, mem in members.items():
        e = ext.get(uid, {})
        confident = True
        if uid in texts:
            title, topics = e.get("title", ""), e.get("topics", [])
            mins = int(e.get("total_minutes") or 0)
            confident = bool(e.get("confident", False)) if e else False
        else:
            title, topics, mins = "", ["no post"], 0
        if uid == ROLE_MODEL_ID:
            title, topics, mins, confident = "", [], ROLE_MODEL_MIN, True  # always 10h
        mins = max(0, min(mins, 1440))
        entries.append(dict(uid=uid, member=mem, name=mem.display_name, minutes=mins,
                            title=title, topics=topics, passed=mins >= BASELINE,
                            confident=confident))
    entries.sort(key=lambda e: -e["minutes"])
    return entries

# ---------- IMAGE ----------
BG_SVG = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='60' height='60' viewBox='0 0 60 60'><path d='M-15,45 L45,-15 M0,60 L60,0 M15,75 L75,15' stroke='%23ff2a2a' stroke-width='1.2' opacity='0.12'/><circle cx='30' cy='30' r='1.5' fill='%23ff2a2a' opacity='0.3'/></svg>"
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
.empty{font-size:30px;font-weight:700;letter-spacing:3px;text-align:center;color:#ff5252;justify-content:center}
.footer{font-family:'Orbitron',sans-serif;font-size:24px;letter-spacing:8px;color:#fff;font-weight:700;
text-shadow:0 0 10px rgba(255,255,255,0.5)}
</style></head><body><div id="capture-area">
<div class="header"><h1 class="title-main">THE GRIND</h1><h2 class="title-sub">LEADERBOARD</h2></div>
<div class="leaderboard-container">""".replace("__BG__", BG_SVG)

async def avatar_uri(member: discord.Member):
    data = await member.display_avatar.replace(size=128, format="png").read()
    return "data:image/png;base64," + base64.b64encode(data).decode()

BROWSER_ARGS = ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
                "--disable-extensions", "--disable-background-networking"]

async def render(passed_entries, days_left: int) -> bytes:
    cards = ""
    for i, e in enumerate(passed_entries[:5]):  # image: only baseline-clearers, max 5
        av = await avatar_uri(e["member"])
        h, m = divmod(e["minutes"], 60)
        cards += (f'<div class="card"><div class="rank" style="color:{RANK_C[i]}">#{i+1}</div>'
                  f'<img class="profile-pic" src="{av}" alt="">'
                  f'<div class="name" style="color:{NAME_C[i]}">{htmllib.escape(e["name"].upper())}</div>'
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
                pass  # fonts slow? screenshot anyway with fallback fonts
            png = await pg.locator("#capture-area").screenshot()
        finally:
            await browser.close()
    return png

# ---------- TEXT ----------
def ordinal(n):
    suf = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"

def fmt(m): return f"{m // 60:02d}h {m % 60:02d}m"
def fmt_short(m): return f"{m // 60}h" if m % 60 == 0 else f"{m // 60}h {m % 60}m"

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

async def build(guild, day: dt.date, end: dt.datetime):
    entries = await collect(guild, day, end)
    caps = await write_captions(entries)
    days_left = max(0, (EXAM_DATE - day).days)

    passed = [e for e in entries if e["passed"]]
    failed = [e for e in entries if not e["passed"]]
    png = await render(passed, days_left)  # image shows only those who cleared the baseline

    def line(e):
        if e["uid"] == ROLE_MODEL_ID:
            cap = ROLE_MODEL_CAPTION
        else:
            cap = clean_caption(caps.get(e["uid"], "📝⚡"))
        mark = "✅" if e["passed"] else "❌"
        return f"{mark} **{escape_markdown(e['name'])}** (**{fmt(e['minutes'])}**): *{cap}*"

    lines = [f"🔥 _*THE GRIND*_ 🔥 — {ordinal(day.day)} {day.strftime('%b')}",
             f"⏳ {days_left} DAYS MORE!", ""]
    lines += [line(e) for e in passed]
    if passed and failed:
        lines.append("-")  # separator between passed and failed
    lines += [line(e) for e in failed]
    lines += ["", f"_*{fmt_short(BASELINE)} baseline. {len(passed)} passed, {len(failed)} failed.*_"]
    shaky = [escape_markdown(e["name"]) for e in entries if not e["confident"]]
    if shaky:
        lines.append(f"⚠️ Couldn't read these confidently, check manually: {', '.join(shaky)}")
    return png, "\n".join(lines)

async def post_board(guild, png: bytes, text: str, prefix: str = ""):
    ch = guild.get_channel(BOARD_CH) or await guild.fetch_channel(BOARD_CH)
    # 1) image first, 2) captions after
    await ch.send(content=prefix or None, file=discord.File(io.BytesIO(png), "leaderboard.png"))
    for part in chunk(text):
        await ch.send(part)
    return ch

# ---------- KEEP-ALIVE (backup; also use UptimeRobot) ----------
async def keepalive():
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        return
    async with aiohttp.ClientSession() as s:
        while True:
            await asyncio.sleep(600)  # every 10 minutes
            try:
                await s.get(url, timeout=aiohttp.ClientTimeout(total=15))
            except Exception:
                pass

# ---------- BOT ----------
intents = discord.Intents.default()
intents.members = True
intents.message_content = True

class Bot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        g = discord.Object(GUILD_ID)
        self.tree.copy_global_to(guild=g)
        await self.tree.sync(guild=g)
        app = web.Application()
        app.router.add_get("/", lambda r: web.Response(text="ok"))
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", int(os.environ.get("PORT", 10000))).start()
        asyncio.create_task(nightly())
        asyncio.create_task(keepalive())

bot = Bot()

@bot.tree.command(name="testnow", description="Generate the leaderboard now (posts in the leaderboard channel)")
@app_commands.default_permissions(administrator=True)
@app_commands.guild_only()
async def testnow(inter: discord.Interaction):
    await inter.response.defer(ephemeral=True)
    if RUN_LOCK.locked():
        await inter.followup.send("⏳ A leaderboard is already being generated. Wait for it to finish.", ephemeral=True)
        return
    async with RUN_LOCK:
        try:
            now = dt.datetime.now(TZ)
            png, text = await build(inter.guild, now.date(), now)
            ch = await post_board(inter.guild, png, text, prefix="🧪 **TEST RUN**")
            await inter.followup.send(f"✅ Posted in {ch.mention}", ephemeral=True)
        except Exception as ex:
            traceback.print_exc()
            await inter.followup.send(f"⚠️ Failed: `{ex}`", ephemeral=True)

async def nightly():
    await bot.wait_until_ready()
    guild = bot.get_guild(GUILD_ID)
    while True:
        now = dt.datetime.now(TZ)
        run_at = now.replace(hour=23, minute=55, second=0, microsecond=0)
        if now >= run_at:
            run_at += dt.timedelta(days=1)
        await asyncio.sleep((run_at - now).total_seconds())
        day = run_at.date()
        payload, err = None, None
        async with RUN_LOCK:
            try:
                payload = await build(guild, day, run_at)
            except Exception as ex:
                traceback.print_exc()
                err = ex
        post_at = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(0, 1), TZ)
        await asyncio.sleep(max(0, (post_at - dt.datetime.now(TZ)).total_seconds()))
        try:
            if payload:
                await post_board(guild, *payload)
            else:
                ch = guild.get_channel(BOARD_CH) or await guild.fetch_channel(BOARD_CH)
                await ch.send(f"⚠️ Leaderboard failed tonight: `{err}`")
        except Exception:
            traceback.print_exc()

bot.run(TOKEN)