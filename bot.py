import os, io, json, base64, asyncio, traceback, html as htmllib
import datetime as dt
from zoneinfo import ZoneInfo

import discord
from discord import app_commands
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
ROLE_MODEL_MIN = 600
ROLE_MODEL_CAPTION = "Catch me if u can! 🏎️💨"
BASELINE = int(os.environ.get("BASELINE_MINUTES", "240"))
EXAM_DATE = dt.date.fromisoformat(os.environ.get("EXAM_DATE", "2027-08-03"))
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ai = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Add 10-15 more of YOUR best captions here. This is what makes the slang feel real.
CAPTION_EXAMPLES = [
    "Block set එකම අතුගාලා 6h+ ගැහුවා! 📝⚡",
    "Tharukaගේ B'day එක අස්සේ 4h+ baseline goda! 📝⚡",
    "14h ගැහුවට පස්සේ adath full afk! 💀💤",
]

# ---------- AI ----------
async def ask_json(prompt: str, temperature: float):
    def call():
        r = ai.models.generate_content(
            model=MODEL, contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=temperature),
        )
        return json.loads(r.text)
    return await asyncio.to_thread(call)

EXTRACT_PROMPT = """You read daily study-progress messages posted by Sri Lankan A/L students.
For each message return the TOTAL study time that was actually completed.
Rules:
- If the message states a total (e.g. "Σt = 6hrs 41mins", "Total - 1 hr 25 mins"), use it.
- Otherwise add up the completed items. Skip items marked ❌ or clearly not done.
- Ignore goals/headers like "8hrs+ today?" or a date such as "03.11". Those are NOT study time.
- Text may use fancy unicode fonts (𝓱𝓻 𝓶𝓲𝓷𝓼). Read them normally.
- Units: hr/hrs/h = hours, min/mins/m = minutes.
- confident=false if you had to guess or the message has no usable time.
Return ONLY a JSON list: [{"id": "...", "total_minutes": int, "summary": "max 6 words, topics studied", "confident": bool}]
Messages:
"""

CAPTION_PROMPT = """You write one-line captions for a Sri Lankan A/L study-group leaderboard, roasting/hyping friends.
Style rules:
- Natural Sri Lankan teen Singlish: Sinhala script for Sinhala words mixed with English study terms. Like a friend texting in a group chat.
- NEVER formal or textbook Sinhala. Never robotic, never a translation of English.
- Max ~12 words. Reference their real topics/time from the data when it helps.
- passed=true: hype and respect. passed=false: light savage roast.
- End with exactly two emojis. No hashtags, no quotes.
- Vary the wording. Do not copy the examples, only match their tone.
Examples of the tone:
{examples}

Return ONLY a JSON list: [{{"id": "...", "caption": "..."}}]
Data (baseline_minutes is the pass mark):
"""

async def extract(texts: dict):
    if not texts:
        return {}
    payload = [{"id": str(k), "text": v} for k, v in texts.items()]
    out = await ask_json(EXTRACT_PROMPT + json.dumps(payload, ensure_ascii=False), 0.0)
    return {int(o["id"]): o for o in out}

async def write_captions(entries):
    rows = [{"id": str(e["uid"]), "name": e["name"], "minutes": e["minutes"],
             "passed": e["passed"], "summary": e["summary"], "baseline_minutes": BASELINE}
            for e in entries if e["uid"] != ROLE_MODEL_ID]
    if not rows:
        return {}
    prompt = CAPTION_PROMPT.format(examples="\n".join(f"- {c}" for c in CAPTION_EXAMPLES))
    try:
        out = await ask_json(prompt + json.dumps(rows, ensure_ascii=False), 1.0)
        return {int(o["id"]): o["caption"].strip() for o in out}
    except Exception:
        traceback.print_exc()
        return {}

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
        confident = True
        if uid == ROLE_MODEL_ID:
            mins, summary = ROLE_MODEL_MIN, ""
        elif uid in texts:
            e = ext.get(uid, {})
            mins = int(e.get("total_minutes") or 0)
            summary = e.get("summary", "")
            confident = bool(e.get("confident", False)) if e else False
        else:
            mins, summary = 0, "no post"
        mins = max(0, min(mins, 1440))
        entries.append(dict(uid=uid, member=mem, name=mem.display_name, minutes=mins,
                            summary=summary, passed=mins >= BASELINE, confident=confident))
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
.name{font-size:32px;font-weight:700;letter-spacing:3px;flex-grow:1}
.time{font-size:32px;font-weight:700;color:#ffca28;letter-spacing:2px}
.footer{font-family:'Orbitron',sans-serif;font-size:24px;letter-spacing:8px;color:#fff;font-weight:700;
text-shadow:0 0 10px rgba(255,255,255,0.5)}
</style></head><body><div id="capture-area">
<div class="header"><h1 class="title-main">THE GRIND</h1><h2 class="title-sub">LEADERBOARD</h2></div>
<div class="leaderboard-container">""".replace("__BG__", BG_SVG)

async def avatar_uri(member: discord.Member):
    data = await member.display_avatar.replace(size=128, format="png").read()
    return "data:image/png;base64," + base64.b64encode(data).decode()

async def render(entries, days_left: int) -> bytes:
    cards = ""
    for i, e in enumerate(entries[:5]):
        av = await avatar_uri(e["member"])
        h, m = divmod(e["minutes"], 60)
        cards += (f'<div class="card"><div class="rank" style="color:{RANK_C[i]}">#{i+1}</div>'
                  f'<img class="profile-pic" src="{av}" alt="">'
                  f'<div class="name" style="color:{NAME_C[i]}">{htmllib.escape(e["name"].upper())}</div>'
                  f'<div class="time">{h:02d}H {m:02d}M</div></div>')
    page = HEAD + cards + f'</div><div class="footer">{days_left} DAYS MORE!</div></div></body></html>'
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        pg = await browser.new_page(viewport={"width": 1200, "height": 1200})
        await pg.set_content(page, wait_until="networkidle")
        await pg.evaluate("document.fonts.ready")
        png = await pg.locator("#capture-area").screenshot()
        await browser.close()
    return png

# ---------- TEXT ----------
def ordinal(n):
    suf = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"

def fmt(m): return f"{m // 60:02d}h {m % 60:02d}m"
def fmt_short(m): return f"{m // 60}h" if m % 60 == 0 else f"{m // 60}h {m % 60}m"

async def build(guild, day: dt.date, end: dt.datetime):
    entries = await collect(guild, day, end)
    caps = await write_captions(entries)
    days_left = max(0, (EXAM_DATE - day).days)
    png = await render(entries, days_left)

    lines = [f"🔥 _*THE GRIND*_ 🔥 — {ordinal(day.day)} {day.strftime('%b')}",
             f"⏳ {days_left} DAYS MORE!", ""]
    for e in entries:
        cap = ROLE_MODEL_CAPTION if e["uid"] == ROLE_MODEL_ID else caps.get(e["uid"], "📝⚡")
        mark = "✅" if e["passed"] else "❌"
        lines.append(f"{mark} _*{e['name']}*_ (_{fmt(e['minutes'])}_): {cap}")
    p = sum(e["passed"] for e in entries)
    lines += ["", f"_*{fmt_short(BASELINE)} baseline. {p} passed, {len(entries) - p} failed.*_"]
    shaky = [e["name"] for e in entries if not e["confident"]]
    if shaky:
        lines.append(f"⚠️ Couldn't read these confidently, check manually: {', '.join(shaky)}")
    return png, "\n".join(lines)

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
        self.loop.create_task(nightly())

bot = Bot()

@bot.tree.command(name="testnow", description="Generate the leaderboard right now (test)")
@app_commands.default_permissions(administrator=True)
@app_commands.guild_only()
async def testnow(inter: discord.Interaction):
    await inter.response.defer()
    try:
        now = dt.datetime.now(TZ)
        png, text = await build(inter.guild, now.date(), now)
        await inter.followup.send(content="🧪 **TEST RUN**\n" + text,
                                  file=discord.File(io.BytesIO(png), "leaderboard.png"))
    except Exception as ex:
        traceback.print_exc()
        await inter.followup.send(f"⚠️ Failed: `{ex}`")

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
        try:
            payload = await build(guild, day, run_at)
        except Exception as ex:
            traceback.print_exc()
            err = ex
        post_at = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(0, 1), TZ)
        await asyncio.sleep(max(0, (post_at - dt.datetime.now(TZ)).total_seconds()))
        ch = guild.get_channel(BOARD_CH) or await guild.fetch_channel(BOARD_CH)
        if payload:
            png, text = payload
            await ch.send(content=text, file=discord.File(io.BytesIO(png), "leaderboard.png"))
        else:
            await ch.send(f"⚠️ Leaderboard failed tonight: `{err}`")

bot.run(TOKEN)