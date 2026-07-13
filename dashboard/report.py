"""
Research Dashboard — kampanya sonuçlarından tek dosyalık statik HTML üretir.

Sunucu/derleme yok: research_memory.sqlite + holdout_audit.sqlite okunur,
kendi kendine yeten (inline CSS/SVG) bir dashboard.html yazılır. Her bölümün
başında Türkçe başlık + kısa açıklama vardır; üstte anlatısal bir özet cümle.
"""
from __future__ import annotations

import html
import os
import sqlite3
from datetime import datetime

from evaluation import build_report
from memory import MemoryStore

# Pipeline aşamaları — funnel sırası (üstten alta)
_STAGE_ORDER = [
    ("compile_error", "Derleme hatası"),
    ("static_rejected", "Sızıntı / statik red"),
    ("critic_rejected", "Critic reddi (ekonomik)"),
    ("duplicate", "Tekrar (novelty)"),
    ("gate_rejected", "Performans kapısı reddi"),
    ("robustness_rejected", "Sağlamlık testi reddi"),
    ("accepted", "KABUL"),
]
_REJECT_STAGES = {"compile_error", "static_rejected", "critic_rejected",
                  "gate_rejected", "robustness_rejected"}

_CSS = """
:root { --bg:#0f1117; --card:#1a1d27; --border:#2a2f3d; --fg:#e6e8ee;
  --muted:#8a90a2; --accent:#5b8def; --good:#3fb950; --bad:#f85149; --warn:#d29922; }
@media (prefers-color-scheme: light) {
  :root { --bg:#f6f7f9; --card:#fff; --border:#e2e5ea; --fg:#1a1d27;
    --muted:#5c6472; --accent:#2f6bd8; --good:#1a7f37; --bad:#cf222e; --warn:#9a6700; } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--fg);
  font-family:-apple-system,Segoe UI,Roboto,sans-serif; line-height:1.55; }
.wrap { max-width:1080px; margin:0 auto; padding:36px 22px 72px; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-.01em; }
.lead { color:var(--muted); font-size:13px; margin:0 0 22px; }
.banner { background:linear-gradient(90deg,rgba(91,141,239,.14),transparent);
  border:1px solid var(--border); border-left:3px solid var(--accent);
  border-radius:10px; padding:14px 18px; font-size:15px; margin-bottom:26px; }
.banner b { color:var(--fg); } .banner .hl { color:var(--accent); font-weight:700; }
section { margin-top:34px; }
h2 { font-size:15px; margin:0 0 3px; letter-spacing:.02em; }
.desc { color:var(--muted); font-size:12.5px; margin:0 0 12px; max-width:760px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.tile { background:var(--card); border:1px solid var(--border); border-radius:10px;
  padding:16px; border-top:3px solid var(--border); }
.tile.g { border-top-color:var(--good); } .tile.r { border-top-color:var(--bad); }
.tile.b { border-top-color:var(--accent); } .tile.w { border-top-color:var(--warn); }
.tile .n { font-size:30px; font-weight:700; line-height:1; }
.tile .l { color:var(--muted); font-size:12px; margin-top:6px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px;
  padding:16px 18px; overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th,td { text-align:left; padding:9px 10px; border-bottom:1px solid var(--border); white-space:nowrap; }
tr:last-child td { border-bottom:none; }
th { color:var(--muted); font-weight:600; font-size:12px; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.bar { height:22px; border-radius:5px; background:var(--accent); min-width:3px; }
.bar.good { background:var(--good); } .bar.bad { background:var(--bad); }
.frow { display:flex; align-items:center; gap:12px; margin:7px 0; }
.frow .lbl { width:190px; font-size:13px; } .frow .cnt { width:34px; text-align:right;
  color:var(--muted); font-variant-numeric:tabular-nums; font-weight:600; }
.frow .track { flex:1; background:var(--border); border-radius:5px; }
.pill { padding:2px 9px; border-radius:20px; font-size:11px; font-weight:700; }
.pill.good { background:rgba(63,185,80,.16); color:var(--good); }
.pill.bad { background:rgba(248,81,73,.16); color:var(--bad); }
.pill.muted { background:var(--border); color:var(--muted); }
.foot { color:var(--muted); font-size:12px; margin-top:44px;
  border-top:1px solid var(--border); padding-top:16px; }
"""


def _q(conn, sql, params=()):
    return conn.execute(sql, params).fetchall()


def _esc(x) -> str:
    return html.escape(str(x))


def _section(title: str, desc: str, body: str) -> str:
    return f'<section><h2>{_esc(title)}</h2><div class="desc">{_esc(desc)}</div>{body}</section>'


def _holdout_counts(holdout_db: str) -> tuple[int, int]:
    if not os.path.exists(holdout_db):
        return (0, 0)
    conn = sqlite3.connect(holdout_db)
    rows = _q(conn, "SELECT passed FROM holdout_access")
    conn.close()
    return (sum(1 for (p,) in rows if p), len(rows))


def _banner(conn, holdout_db: str) -> str:
    total = _q(conn, "SELECT COUNT(*) FROM experiment")[0][0]
    acc = _q(conn, "SELECT COUNT(*) FROM experiment WHERE decision='accept'")[0][0]
    passed, cand = _holdout_counts(holdout_db)
    hold_txt = (f"bunlardan <span class='hl'>{passed}</span> tanesi kilitli holdout "
                f"dönemini geçti" if cand else "holdout değerlendirmesi henüz yapılmadı")
    return (f'<div class="banner">Bu kampanyada LLM otonom olarak '
            f'<b>{total}</b> hipotez üretip test etti; '
            f'<span class="hl">{acc}</span> tanesi tüm süzgeçlerden geçip kabul edildi, '
            f'{hold_txt}. Aşağıdaki her bölüm sürecin bir yönünü gösterir.</div>')


def _tiles(conn) -> str:
    total = _q(conn, "SELECT COUNT(*) FROM experiment")[0][0]
    by_dec = dict(_q(conn, "SELECT decision, COUNT(*) FROM experiment GROUP BY decision"))
    fams = _q(conn, "SELECT COUNT(DISTINCT family) FROM experiment WHERE sharpe IS NOT NULL")[0][0]
    items = [("Toplam hipotez", total, "b"), ("Kabul edilen", by_dec.get("accept", 0), "g"),
             ("Reddedilen", by_dec.get("reject", 0), "r"),
             ("Tekrar (elendi)", by_dec.get("duplicate", 0), "w"),
             ("Denenen aile", fams, "b")]
    return '<div class="tiles">' + "".join(
        f'<div class="tile {c}"><div class="n">{v}</div><div class="l">{_esc(l)}</div></div>'
        for l, v, c in items) + "</div>"


def _funnel(conn) -> str:
    counts = dict(_q(conn, "SELECT stage, COUNT(*) FROM experiment GROUP BY stage"))
    mx = max(list(counts.values()) + [1])
    rows = []
    for stage, label in _STAGE_ORDER:
        c = counts.get(stage, 0)
        w = int(100 * c / mx)
        cls = "good" if stage == "accepted" else ("bad" if c and stage in _REJECT_STAGES else "")
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
        return '<div class="card desc">Bu kampanyada kabul edilen strateji olmadı.</div>'
    body = "".join(
        f"<tr><td>{_esc(h)}</td><td>{_esc(t)}</td>"
        f'<td class="num">{s:.2f}</td><td class="num">%{(d or 0)*100:.0f}</td></tr>'
        for h, t, s, d in rows)
    return ('<div class="card"><table><tr><th>Kimlik</th><th>Strateji</th>'
            '<th>Sharpe</th><th>Maks. düşüş</th></tr>' + body + "</table></div>")


def _multiple_testing(memory_db: str) -> str:
    store = MemoryStore(memory_db)
    rows = build_report(store.backtested_experiments())
    store.close()
    if not rows:
        return '<div class="card desc">Backtest edilen deney yok.</div>'
    body = ""
    for r in rows:
        fdr = ('<span class="pill good">GEÇTİ</span>' if r.survives_fdr
               else '<span class="pill muted">geçmedi</span>')
        dsr = f"{r.dsr:.2f}" + (" ★" if r.dsr > 0.95 else "")
        body += (f"<tr><td>{_esc(r.hypothesis_id)}</td>"
                 f'<td class="num">{r.ann_sharpe:.2f}</td>'
                 f'<td class="num">{r.raw_p:.3f}</td><td class="num">{dsr}</td>'
                 f'<td class="num">[{r.ci_low:.2f}, {r.ci_high:.2f}]</td>'
                 f"<td>{fdr}</td></tr>")
    return ('<div class="card"><table><tr><th>Kimlik</th><th>Sharpe</th>'
            '<th>ham p</th><th>DSR</th><th>%95 güven aralığı</th><th>FDR</th></tr>'
            + body + "</table></div>"
            '<div class="desc" style="margin-top:8px">★ = DSR &gt; 0.95: deneme sayısı '
            'düzeltildikten sonra bile anlamlı. Güven aralığı sıfırı içeriyorsa sonuç '
            'kesin değildir.</div>')


def _holdout(holdout_db: str) -> str:
    if not os.path.exists(holdout_db):
        return '<div class="card desc">Holdout değerlendirmesi yapılmadı.</div>'
    conn = sqlite3.connect(holdout_db)
    rows = _q(conn, "SELECT hypothesis_id, sharpe, passed FROM holdout_access ORDER BY sharpe DESC")
    conn.close()
    if not rows:
        return '<div class="card desc">Holdout adayı yok.</div>'

    def _pill(passed) -> str:
        return ('<span class="pill good">GEÇTİ</span>' if passed
                else '<span class="pill bad">KALDI</span>')

    body = "".join(
        f'<tr><td>{_esc(h)}</td><td class="num">{s:.2f}</td><td>{_pill(p)}</td></tr>'
        for h, s, p in rows)
    return ('<div class="card"><table><tr><th>Kimlik</th><th>Holdout Sharpe</th>'
            '<th>Sonuç</th></tr>' + body + "</table></div>")


def _families(conn) -> str:
    rows = _q(conn, """SELECT family,
                         SUM(CASE WHEN decision='accept' THEN 1 ELSE 0 END),
                         COUNT(*)
                       FROM experiment WHERE sharpe IS NOT NULL GROUP BY family
                       ORDER BY 3 DESC""")
    if not rows:
        return '<div class="card desc">Backtest edilen aile yok.</div>'
    mx = max([t for _, _, t in rows] + [1])
    out = []
    for fam, acc, tot in rows:
        w = int(100 * tot / mx)
        out.append(
            f'<div class="frow"><div class="lbl">{_esc(fam)}</div>'
            f'<div class="track"><div class="bar" style="width:{w}%"></div></div>'
            f'<div class="cnt">{acc}/{tot}</div></div>')
    return '<div class="card">' + "".join(out) + "</div>"


def generate_dashboard(memory_db: str, holdout_db: str, out_path: str,
                       campaign_name: str = "") -> str:
    conn = sqlite3.connect(memory_db)
    ts = datetime.now().strftime("%d.%m.%Y %H:%M")
    parts = [
        f'<h1>Araştırma Paneli</h1>',
        f'<p class="lead">Kampanya: <b>{_esc(campaign_name)}</b> · Oluşturma: {ts}</p>',
        _banner(conn, holdout_db),
        _section("Kampanya Özeti",
                 "Bu turda üretilen hipotezlerin karar dağılımı.",
                 _tiles(conn)),
        _section("Araştırma Hunisi — Hipotezler Nerede Elendi?",
                 "Her hipotez soldan sağa bu aşamalardan geçer; bir aşamada elenirse "
                 "orada durur. Kırmızı = elendi, mavi = tekrar, yeşil = kabul. "
                 "Sağdaki sayı o aşamada sonlanan hipotez adedidir.",
                 _funnel(conn)),
        _section("En İyi Stratejiler",
                 "Tüm süzgeçlerden geçip kabul edilen stratejiler, araştırma dönemi "
                 "Sharpe oranına göre sıralı.",
                 _leaderboard(conn)),
        _section("Çoklu Test Düzeltmesi — 'Kabul' ≠ 'İstatistiksel Geçerli'",
                 "Çok sayıda deneme yapıldığında yüksek bir Sharpe tesadüfen çıkabilir. "
                 "Deflated Sharpe (DSR) ve FDR bunu düzeltir: FDR 'GEÇTİ' değilse sonuç "
                 "istatistiksel olarak kanıtlanmış sayılmaz.",
                 _multiple_testing(memory_db)),
        _section("Kilitli Dönem Sınavı (Holdout)",
                 "Araştırma sırasında hiç görülmeyen, kilitli bir dönemde yapılan son "
                 "test. Bir stratejinin gerçekten genelleyip genellemediği buradan "
                 "anlaşılır (araştırma ajanı bu veriye asla erişemez).",
                 _holdout(holdout_db)),
        _section("Aile Performansı — Bütçe Dağılımı",
                 "Her strateji ailesinin kabul/toplam oranı. Sistem araştırma bütçesini "
                 "başarılı ailelere Thompson sampling (bandit) ile kaydırır.",
                 _families(conn)),
    ]
    conn.close()
    doc = (f'<!doctype html><html lang="tr"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width, initial-scale=1">'
           f'<title>Araştırma Paneli — {_esc(campaign_name)}</title>'
           f'<style>{_CSS}</style></head><body><div class="wrap">'
           + "".join(parts)
           + '<div class="foot">LLM Tabanlı Otonom Quant Araştırmacısı · '
             'otomatik üretilmiş rapor</div></div></body></html>')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = generate_dashboard(
        os.path.join(here, "research_memory.sqlite"),
        os.path.join(here, "holdout_audit.sqlite"),
        os.path.join(here, "dashboard.html"),
        campaign_name="skeleton_daily_alpha_v0")
    print(f"Dashboard yazıldı: {out}")
