"""
Research Dashboard — kampanya sonuçlarından tek dosyalık statik HTML üretir.

Sunucu/derleme yok: research_memory.sqlite + holdout_audit.sqlite okunur,
kendi kendine yeten (inline CSS/SVG) bir dashboard.html yazılır. Bölümler:
  - Özet istatistikler + pipeline funnel (hipotezler nerede eleniyor)
  - Leaderboard (kabul edilenler)
  - Multiple testing raporu (DSR/FDR)
  - Holdout sonuçları
  - Aile performansı (bandit görünümü)
"""
from __future__ import annotations

import html
import os
import sqlite3
from datetime import datetime

from evaluation import build_report
from memory import MemoryStore

# Pipeline aşamaları — funnel sırası (üstten alta daralır)
_STAGE_ORDER = [
    ("compile_error", "Derleme hatası"),
    ("static_rejected", "Sızıntı/statik red"),
    ("critic_rejected", "Critic reddi"),
    ("duplicate", "Tekrar (novelty)"),
    ("gate_rejected", "Hard gate reddi"),
    ("robustness_rejected", "Sağlamlık reddi"),
    ("accepted", "KABUL"),
]

_CSS = """
:root { --bg:#0f1117; --card:#1a1d27; --border:#2a2f3d; --fg:#e6e8ee;
  --muted:#8a90a2; --accent:#5b8def; --good:#3fb950; --bad:#f85149; --warn:#d29922; }
@media (prefers-color-scheme: light) {
  :root { --bg:#f6f7f9; --card:#fff; --border:#e2e5ea; --fg:#1a1d27;
    --muted:#6a7080; --accent:#2f6bd8; --good:#1a7f37; --bad:#cf222e; --warn:#9a6700; } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--fg);
  font-family:-apple-system,Segoe UI,Roboto,sans-serif; line-height:1.5; }
.wrap { max-width:1100px; margin:0 auto; padding:32px 20px 64px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:16px; margin:32px 0 12px;
  text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.sub { color:var(--muted); font-size:13px; margin-bottom:8px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; }
.tile { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:16px; }
.tile .n { font-size:28px; font-weight:700; } .tile .l { color:var(--muted); font-size:12px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px;
  padding:16px 18px; overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th,td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--border); white-space:nowrap; }
th { color:var(--muted); font-weight:600; } td.num { text-align:right; font-variant-numeric:tabular-nums; }
.bar { height:22px; border-radius:5px; background:var(--accent); min-width:2px; }
.bar.good { background:var(--good); } .bar.bad { background:var(--bad); }
.frow { display:flex; align-items:center; gap:10px; margin:6px 0; }
.frow .lbl { width:180px; font-size:13px; } .frow .cnt { width:36px; text-align:right;
  color:var(--muted); font-variant-numeric:tabular-nums; }
.frow .track { flex:1; background:var(--border); border-radius:5px; }
.pill { padding:2px 8px; border-radius:20px; font-size:11px; font-weight:600; }
.pill.good { background:rgba(63,185,80,.15); color:var(--good); }
.pill.bad { background:rgba(248,81,73,.15); color:var(--bad); }
.foot { color:var(--muted); font-size:12px; margin-top:40px; }
"""


def _q(conn, sql, params=()):
    return conn.execute(sql, params).fetchall()


def _esc(x) -> str:
    return html.escape(str(x))


def _tiles(conn) -> str:
    total = _q(conn, "SELECT COUNT(*) FROM experiment")[0][0]
    by_dec = dict(_q(conn, "SELECT decision, COUNT(*) FROM experiment GROUP BY decision"))
    fams = _q(conn, "SELECT COUNT(DISTINCT family) FROM experiment WHERE sharpe IS NOT NULL")[0][0]
    items = [("Toplam deney", total), ("Kabul", by_dec.get("accept", 0)),
             ("Red", by_dec.get("reject", 0)), ("Tekrar", by_dec.get("duplicate", 0)),
             ("Denenen aile", fams)]
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="n">{v}</div><div class="l">{_esc(l)}</div></div>'
        for l, v in items) + "</div>"


_REJECT_STAGES = {"compile_error", "static_rejected", "critic_rejected",
                  "gate_rejected", "robustness_rejected"}


def _funnel(conn) -> str:
    counts = dict(_q(conn, "SELECT stage, COUNT(*) FROM experiment GROUP BY stage"))
    mx = max([c for c in counts.values()] + [1])
    rows = []
    for stage, label in _STAGE_ORDER:
        c = counts.get(stage, 0)
        w = int(100 * c / mx)
        if stage == "accepted":
            cls = "good"
        elif c and stage in _REJECT_STAGES:
            cls = "bad"
        else:
            cls = ""
        rows.append(
            f'<div class="frow"><div class="lbl">{_esc(label)}</div>'
            f'<div class="track"><div class="bar {cls}" style="width:{w}%"></div></div>'
            f'<div class="cnt">{c}</div></div>')
    return '<div class="card">' + "".join(rows) + "</div>"


def _leaderboard(conn) -> str:
    rows = _q(conn, """SELECT hypothesis_id, title, sharpe, max_drawdown FROM experiment
                       WHERE decision='accept' AND sharpe IS NOT NULL
                       ORDER BY sharpe DESC LIMIT 20""")
    if not rows:
        return '<div class="card sub">Kabul edilen strateji yok.</div>'
    body = "".join(
        f"<tr><td>{_esc(h)}</td><td>{_esc(t)}</td>"
        f'<td class="num">{s:.2f}</td><td class="num">%{(d or 0)*100:.0f}</td></tr>'
        for h, t, s, d in rows)
    return ('<div class="card"><table><tr><th>ID</th><th>Başlık</th>'
            '<th>Sharpe</th><th>Max DD</th></tr>' + body + "</table></div>")


def _multiple_testing(memory_db: str) -> str:
    store = MemoryStore(memory_db)
    rows = build_report(store.backtested_experiments())
    store.close()
    if not rows:
        return '<div class="card sub">Backtest edilen deney yok.</div>'
    body = ""
    for r in rows:
        fdr = '<span class="pill good">GEÇTİ</span>' if r.survives_fdr else "–"
        dsr = f"{r.dsr:.2f}" + (" ★" if r.dsr > 0.95 else "")
        body += (f"<tr><td>{_esc(r.hypothesis_id)}</td>"
                 f'<td class="num">{r.ann_sharpe:.2f}</td>'
                 f'<td class="num">{r.raw_p:.3f}</td><td class="num">{dsr}</td>'
                 f'<td class="num">[{r.ci_low:.2f}, {r.ci_high:.2f}]</td>'
                 f"<td>{fdr}</td></tr>")
    return ('<div class="card"><table><tr><th>ID</th><th>Sharpe</th><th>ham p</th>'
            '<th>DSR</th><th>%95 CI</th><th>FDR</th></tr>' + body + "</table></div>"
            '<div class="sub">★ = DSR&gt;0.95 (deneme sayısı düzeltilse bile anlamlı)</div>')


def _holdout(holdout_db: str) -> str:
    if not os.path.exists(holdout_db):
        return '<div class="card sub">Holdout değerlendirmesi yapılmadı.</div>'
    conn = sqlite3.connect(holdout_db)
    rows = _q(conn, "SELECT hypothesis_id, sharpe, passed FROM holdout_access ORDER BY sharpe DESC")
    conn.close()
    if not rows:
        return '<div class="card sub">Holdout adayı yok.</div>'

    def _pill(passed) -> str:
        return ('<span class="pill good">GEÇTİ</span>' if passed
                else '<span class="pill bad">KALDI</span>')

    body = "".join(
        f'<tr><td>{_esc(h)}</td><td class="num">{s:.2f}</td><td>{_pill(p)}</td></tr>'
        for h, s, p in rows)
    return ('<div class="card"><table><tr><th>ID</th><th>Holdout Sharpe</th>'
            '<th>Sonuç</th></tr>' + body + "</table></div>")


def _families(conn) -> str:
    rows = _q(conn, """SELECT family,
                         SUM(CASE WHEN decision='accept' THEN 1 ELSE 0 END),
                         COUNT(*)
                       FROM experiment WHERE sharpe IS NOT NULL GROUP BY family
                       ORDER BY 3 DESC""")
    if not rows:
        return '<div class="card sub">Backtest edilen aile yok.</div>'
    mx = max([t for _, _, t in rows] + [1])
    out = []
    for fam, acc, tot in rows:
        w = int(100 * tot / mx)
        out.append(
            f'<div class="frow"><div class="lbl">{_esc(fam)}</div>'
            f'<div class="track"><div class="bar" style="width:{w}%"></div></div>'
            f'<div class="cnt">{acc}/{tot}</div></div>')
    return '<div class="card">' + "".join(out) + \
           '</div><div class="sub">kabul / toplam backtest (bandit bütçesi buna göre dağılır)</div>'


def generate_dashboard(memory_db: str, holdout_db: str, out_path: str,
                       campaign_name: str = "") -> str:
    conn = sqlite3.connect(memory_db)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    html_doc = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research Dashboard — {_esc(campaign_name)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>Research Dashboard</h1>
<div class="sub">{_esc(campaign_name)} · {ts}</div>
{_tiles(conn)}
<h2>Pipeline Funnel — hipotezler nerede eleniyor</h2>{_funnel(conn)}
<h2>Leaderboard — kabul edilen stratejiler</h2>{_leaderboard(conn)}
<h2>Multiple Testing — "kabul" != "istatistiksel geçerli"</h2>{_multiple_testing(memory_db)}
<h2>Holdout — kilitli dönem, son sınav</h2>{_holdout(holdout_db)}
<h2>Aile Performansı — bandit bütçe görünümü</h2>{_families(conn)}
<div class="foot">LLM Tabanlı Otonom Quant Araştırmacısı · otomatik üretilmiş rapor</div>
</div></body></html>"""
    conn.close()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return out_path


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = generate_dashboard(
        os.path.join(here, "research_memory.sqlite"),
        os.path.join(here, "holdout_audit.sqlite"),
        os.path.join(here, "dashboard.html"),
        campaign_name="skeleton_daily_alpha_v0")
    print(f"Dashboard yazıldı: {out}")
