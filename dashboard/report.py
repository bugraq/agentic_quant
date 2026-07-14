"""
Research Dashboard — kampanya sonuçlarından tek dosyalık statik HTML üretir.

Sunucu/derleme yok: research_memory.sqlite + holdout_audit.sqlite okunur,
kendi kendine yeten (inline CSS/SVG) bir dashboard.html yazılır. Her bölümün
başında Türkçe başlık + kısa açıklama vardır; üstte anlatısal bir özet cümle.
"""
from __future__ import annotations

import html
import json
import os
import sqlite3
from datetime import datetime

from evaluation import build_report, evaluate_strategies
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
.pill.warn { background:rgba(210,153,34,.16); color:var(--warn); }
.pill.muted { background:var(--border); color:var(--muted); }
.detail { background:var(--card); border:1px solid var(--border); border-radius:10px;
  padding:14px 16px; margin-bottom:12px; }
.detail .dh { font-weight:600; margin-bottom:8px; font-size:14px; }
.detail .did { color:var(--accent); font-family:ui-monospace,monospace; margin-right:6px; }
.detail .dsh { float:right; color:var(--good); font-weight:700; font-size:13px; }
.detail .drow { font-size:13px; margin:5px 0; }
.detail .drow b { color:var(--muted); font-weight:600; }
.detail code { background:var(--bg); border:1px solid var(--border); border-radius:5px;
  padding:3px 7px; font-size:12.5px; display:inline-block; margin-top:3px;
  font-family:ui-monospace,monospace; word-break:break-all; }
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
        copy_tag = f" ×{r.n_copies}" if r.n_copies > 1 else ""
        body += (f"<tr><td>{_esc(r.hypothesis_id)}{_esc(copy_tag)}</td>"
                 f'<td class="num">{r.ann_sharpe:.2f}</td>'
                 f'<td class="num">{r.raw_p:.3f}</td><td class="num">{dsr}</td>'
                 f'<td class="num">[{r.ci_low:.2f}, {r.ci_high:.2f}]</td>'
                 f"<td>{fdr}</td></tr>")
    return ('<div class="card"><table><tr><th>Kimlik</th><th>Sharpe</th>'
            '<th>ham p</th><th>DSR</th><th>%95 güven aralığı</th><th>FDR</th></tr>'
            + body + "</table></div>"
            '<div class="desc" style="margin-top:8px">★ = DSR &gt; 0.95: deneme sayısı '
            'düzeltildikten sonra bile anlamlı. Güven aralığı sıfırı içeriyorsa sonuç '
            'kesin değildir. ×N = N deneme birebir aynı getiriyi üretti (ölü parametre: '
            'o pencere stratejiyi etkilemiyor).</div>')


def _pareto(memory_db: str) -> str:
    """Çok amaçlı Pareto sıralaması (Doküman 11.2)."""
    store = MemoryStore(memory_db)
    evals = evaluate_strategies(store.accepted_full())
    store.close()
    if not evals:
        return '<div class="card desc">Değerlendirilecek kabul edilmiş strateji yok.</div>'
    body = ""
    for e in evals:
        star = '<span class="pill good">Pareto-optimal</span>' if e.pareto_optimal else '–'
        body += (f"<tr><td>{_esc(e.hypothesis_id)}</td>"
                 f'<td class="num">{e.sharpe:.2f}</td>'
                 f'<td class="num">{e.sharpe_lb:.2f}</td>'
                 f'<td class="num">%{e.max_drawdown*100:.0f}</td>'
                 f'<td class="num">{e.turnover:.0f}</td>'
                 f'<td class="num">{e.score:.2f}</td><td>{star}</td></tr>')
    return ('<div class="card"><table><tr><th>Kimlik</th><th>Sharpe</th>'
            '<th>Sharpe alt-sınır</th><th>Maks DD</th><th>Turnover</th>'
            '<th>Skor</th><th>Pareto</th></tr>' + body + "</table></div>"
            '<div class="desc" style="margin-top:8px">Skor = Sharpe_alt-sınır '
            '− 0.5·DD − 0.002·turnover (bütçe tahsisi için yardımcı sinyal). '
            'Pareto-optimal = hiçbir stratejiye tüm boyutlarda yenik düşmeyen.</div>')


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


def _signal_formula(e: dict) -> str:
    """DSL sinyal ağacını okunabilir formüle çevir (görsel için)."""
    op = e["op"]
    if op == "field":
        return e.get("field") or "?"
    if op == "const":
        return str(e.get("value"))
    if op == "feature_ref":
        return e.get("name") or "?"
    inner = ", ".join(_signal_formula(i) for i in e.get("inputs", []) if isinstance(i, dict))
    w = f", pencere={e['window']}" if e.get("window") else ""
    return f"{op}({inner}{w})"


# --- Düz Türkçe çeviri (quant olmayan biri okuyunca anlasın) ---------------
_FIELD_TR = {
    "close": "kapanış fiyatı", "open": "açılış fiyatı", "high": "gün-içi en yüksek",
    "low": "gün-içi en düşük", "adjusted_close": "düzeltilmiş kapanış",
    "volume": "işlem hacmi", "dollar_volume": "dolar hacmi", "market_cap": "piyasa değeri",
}


def _describe_node(e: dict, feats: dict) -> str:
    """DSL düğümünü düz Türkçe bir ifadeye çevirir (özyinelemeli)."""
    op = e.get("op")
    if op == "field":
        return _FIELD_TR.get(e.get("field"), e.get("field") or "?")
    if op == "const":
        return str(e.get("value"))
    if op == "feature_ref":
        name = e.get("name") or "?"
        return _describe_node(feats[name], feats) if name in feats else name
    ins = [i for i in e.get("inputs", []) if isinstance(i, dict)]
    a = _describe_node(ins[0], feats) if ins else "?"
    b = _describe_node(ins[1], feats) if len(ins) > 1 else "?"
    w = e.get("window")
    gun = f"{w} günlük" if w else ""
    if op == "return":
        return f"son {gun} getirisi"
    if op in ("rolling_mean", "ewma"):
        return f"{a} — {gun} ortalaması"
    if op in ("rolling_std", "volatility"):
        return f"{a} — {gun} oynaklığı"
    if op == "zscore":
        return f"{a} — {gun} z-skoru (ortalamadan kaç std sapmış)"
    if op == "delta":
        return f"{a} — {gun} değişimi"
    if op in ("rolling_min",):
        return f"{a} — {gun} dip"
    if op in ("rolling_max",):
        return f"{a} — {gun} tepe"
    if op in ("cross_sectional_rank", "quantile", "rolling_rank"):
        return f"{a} sıralaması"
    if op in ("normalize", "demean", "neutralize_market", "neutralize_sector"):
        return f"piyasadan arındırılmış {a}"
    if op == "winsorize":
        return f"aşırı uçları budanmış {a}"
    if op == "negate":
        return f"{a} (tersi — düşük olanı öne alır)"
    if op == "multiply":
        return f"{a} × {b}"
    if op in ("divide", "ratio"):
        return f"{a} / {b}"
    if op == "add":
        return f"{a} + {b}"
    if op == "subtract":
        return f"{a} eksi {b}"
    if op == "greater_than":
        return f"{a} > {b} koşulu"
    if op == "less_than":
        return f"{a} < {b} koşulu"
    if op == "conditional":
        c = _describe_node(ins[2], feats) if len(ins) > 2 else "?"
        return f"eğer {a} ise {b}, değilse {c}"
    if op == "correlation":
        return f"{a} ile {b} korelasyonu ({gun})"
    return f"{op}({a})"


def _plain_strategy(h: dict) -> str:
    """Hipotezi tek cümlelik düz Türkçe stratejiye çevirir (sinyal + portföy)."""
    feats = {f.get("name"): f.get("expression", {})
             for f in h.get("features", []) if isinstance(f, dict)}
    core = _describe_node(h.get("signal", {}), feats)
    ptype = h.get("portfolio", {}).get("type", "")
    if "long_short" in ptype:
        action = ("hisseleri bu değere göre sıralar; en yüksek olanları AL (long), "
                  "en düşük olanları SAT (short)")
    elif "long_only" in ptype:
        action = "en yüksek değere sahip hisseleri AL (sadece long), gerisini alma"
    else:
        action = "hisseleri bu değere göre seçer"
    return f"Her gün her hisse için <b>{core}</b> hesaplanır; sonra {action}."


# İssue tipi -> insan-dostu Türkçe başlık (reddetme nedeni)
_REASON_TR = {
    "compile_error": "Derlenmedi (geçersiz strateji yapısı)",
    "lookahead": "Geleceğe bakma (sızıntı) tespit edildi",
    "leakage": "Veri sızıntısı tespit edildi",
    "disallowed_field": "İzin verilmeyen veri alanı kullandı",
    "disallowed_operator": "İzin verilmeyen operatör kullandı",
    "yapısal_duplicate": "Daha önce denenen bir stratejiyle aynı (yapısal tekrar)",
    "davranışsal_duplicate": "Başka bir stratejiyle neredeyse aynı sinyali üretti (tekrar)",
    "not_robust": "Sağlamlık testlerini geçemedi (şansa/ayara aşırı bağımlı)",
    "claim_signal_mismatch": "İddia ile sinyal uyuşmuyor (critic reddi)",
    "sharpe_below_threshold": "Getiri/risk (Sharpe) eşiğin altında",
    "drawdown_exceeded": "Maksimum düşüş sınırı aşıldı",
    "turnover_exceeded": "İşlem sıklığı (turnover) sınırı aşıldı",
    "insufficient_positive_folds": "Dönemler arası tutarsız (yeterli pozitif fold yok)",
}


def _humanize_issue(issues_json: str | None) -> str:
    """issues_json'daki ilk sorunu insan-dostu bir cümleye çevirir."""
    if not issues_json:
        return "—"
    try:
        issues = json.loads(issues_json)
    except (json.JSONDecodeError, TypeError):
        return "—"
    if not issues:
        return "—"
    it = issues[0]
    typ = it.get("type", "")
    label = _REASON_TR.get(typ)
    desc = it.get("description", "")
    if label:
        return f"{label}" + (f" — {desc}" if desc else "")
    return desc or typ or "—"


def _details(conn) -> str:
    """Hipotez detayı (Doküman 20) — kabul edilen stratejilerin TAM içeriği."""
    rows = _q(conn, """SELECT hypothesis_id, sharpe, hypothesis_json, model_name, prompt_hash, seed
                       FROM experiment
                       WHERE decision='accept' AND hypothesis_json IS NOT NULL
                       ORDER BY sharpe DESC LIMIT 6""")
    if not rows:
        return '<div class="card desc">Detay gösterilecek kabul edilmiş strateji yok.</div>'
    cards = []
    for hid, sharpe, hj, model_name, prompt_hash, seed in rows:
        h = json.loads(hj)
        mech = h.get("economic_mechanism", {})
        fails = mech.get("expected_failure_conditions", []) or []
        f = h.get("falsification", {})
        cards.append(f"""<div class="detail">
  <div class="dh"><span class="did">{_esc(hid)}</span> {_esc(h.get('title',''))}
    <span class="dsh">Sharpe {sharpe:.2f}</span></div>
  <div class="drow"><b>Ne yapıyor (düz anlatım):</b> {_plain_strategy(h)}</div>
  <div class="drow"><b>İddia:</b> {_esc(h.get('claim',''))}</div>
  <div class="drow"><b>Ekonomik mekanizma:</b> {_esc(mech.get('type',''))} — {_esc(mech.get('description',''))}</div>
  <div class="drow"><b>Beklenen başarısızlık koşulları:</b> {_esc(', '.join(fails) or '—')}</div>
  <div class="drow"><b>Aile / portföy:</b> {_esc(h.get('family',''))} · {_esc(h.get('portfolio',{}).get('type',''))}</div>
  <div class="drow"><b>Sinyal (DSL formülü):</b><br><code>{_esc(_signal_formula(h.get('signal',{})))}</code></div>
  <div class="drow"><b>Çürütme eşiği (ön kayıt):</b> min OOS Sharpe {f.get('minimum_oos_sharpe','—')}, maks turnover {f.get('maximum_turnover','—')}, maks DD {f.get('maximum_drawdown','—')}</div>
  <div class="drow"><b>Tekrar-üretilebilirlik:</b> model {_esc(model_name or '—')} · prompt {_esc(prompt_hash or '—')} · seed {_esc(seed if seed is not None else '—')}</div>
</div>""")
    return "".join(cards)


def _all_hypotheses(conn) -> str:
    """Denenen HER hipotez (kabul+red) — düz Türkçe strateji + sonuç + neden.

    Bu bölüm kampanyanın asıl hikâyesidir: LLM ne denedi, ne oldu, NİYE. Kabul
    çıkmasa bile (gerçek veride sık olur) sistemin ne yaptığı buradan anlaşılır.
    """
    rows = _q(conn, """SELECT hypothesis_id, title, family, decision, sharpe,
                              hypothesis_json, issues_json, stage,
                              parent_hypothesis_id
                       FROM experiment ORDER BY id""")
    if not rows:
        return '<div class="card desc">Henüz hipotez üretilmedi.</div>'

    def _pill(dec: str) -> str:
        if dec == "accept":
            return '<span class="pill good">KABUL</span>'
        if dec == "duplicate":
            return '<span class="pill muted">TEKRAR</span>'
        return '<span class="pill bad">RED</span>'

    # Parametre-arama denemeleri LLM hipotezi DEĞİL, optimizer'ın pencere
    # varyantları — detay listesini boğmasınlar; parent başına TEK satıra katla.
    # (Çoklu-test sayımında yine tam olarak yer alırlar.)
    param_counts: dict = {}
    main_rows = []
    for r in rows:
        if r[7] == "parameter_search":
            param_counts[r[8] or "?"] = param_counts.get(r[8] or "?", 0) + 1
        else:
            main_rows.append(r)

    cards = []
    for hid, title, family, dec, sharpe, hj, issues, _stage, _parent in main_rows:
        h = json.loads(hj) if hj else {}
        plain = _plain_strategy(h) if h else "—"
        sh = f' · araştırma Sharpe {sharpe:.2f}' if sharpe is not None else ""
        if dec == "accept":
            reason = "Tüm süzgeçlerden geçti (sızıntı, performans, sağlamlık)."
        else:
            reason = _humanize_issue(issues)
        extra = ""
        if hid in param_counts:
            extra = (f'<div class="drow"><b>Parametre araması:</b> optimizer bu '
                     f'hipotezin pencerelerinde {param_counts[hid]} varyant denedi '
                     f'(hepsi çoklu-test sayımında).</div>')
        cards.append(f"""<div class="detail">
  <div class="dh"><span class="did">{_esc(hid)}</span> {_esc(title or '')}
    <span style="float:right">{_pill(dec)}</span></div>
  <div class="drow"><b>Aile:</b> {_esc(family or '—')}{_esc(sh)}</div>
  <div class="drow"><b>Ne yapıyor:</b> {plain}</div>
  <div class="drow"><b>Sonuç / neden:</b> {_esc(reason)}</div>
  {extra}
</div>""")
    return "".join(cards)


def _lineage(conn) -> str:
    """Hipotez soy ağacı (Doküman 13/20) — parent -> child ilişkileri."""
    rows = _q(conn, """SELECT parent_hypothesis_id, hypothesis_id, relation_type, decision
                       FROM experiment WHERE parent_hypothesis_id IS NOT NULL
                       ORDER BY id""")
    if not rows:
        return '<div class="card desc">Henüz türetilmiş (revision/inversion) hipotez yok.</div>'
    body = "".join(
        f'<tr><td>{_esc(p)}</td><td>→ {_esc(rel or "?")} →</td><td>{_esc(c)}</td>'
        f'<td>{_esc(dec)}</td></tr>'
        for p, c, rel, dec in rows)
    return ('<div class="card"><table><tr><th>Ebeveyn</th><th>İlişki</th>'
            '<th>Türev</th><th>Sonuç</th></tr>' + body + "</table></div>")


def _render_report(rep: dict) -> str:
    """Bir ReviewReport (dict) -> renkli kontrol listesi HTML."""
    pill = {"ok": "good", "warn": "warn", "fail": "bad"}
    label = {"ok": "TEMİZ", "warn": "DİKKAT", "fail": "SORUN"}
    v = rep.get("verdict", "ok")
    head = (f'<div class="drow"><b>{_esc(rep.get("reviewer",""))}:</b> '
            f'<span class="pill {pill.get(v,"muted")}">{label.get(v, v)}</span></div>')
    items = "".join(
        f'<div class="drow" style="margin-left:10px">'
        f'<span class="pill {pill.get(c.get("status"),"muted")}">'
        f'{_esc(c.get("status"))}</span> '
        f'<b>{_esc(c.get("name"))}:</b> {_esc(c.get("detail"))}</div>'
        for c in rep.get("checks", []))
    return head + items


def _reviewers(memory_db: str) -> str:
    """Bağımsız reviewer ajanları (Doküman 15): Backtest Auditor + Statistical Reviewer.

    Auditor raporu kabul sırasında saklanır (reviews_json); Statistical Reviewer
    çoklu-test satırından rapor-zamanı hesaplanır.
    """
    from agents.statistical_reviewer import StatisticalReviewer

    store = MemoryStore(memory_db)
    rows = build_report(store.backtested_experiments())
    store.close()
    stat_by_hid = {r.hypothesis_id: r for r in rows}

    conn = sqlite3.connect(memory_db)
    accepted = _q(conn, """SELECT hypothesis_id, title, reviews_json FROM experiment
                           WHERE decision='accept' ORDER BY sharpe DESC LIMIT 6""")
    conn.close()
    if not accepted:
        return '<div class="card desc">Kabul edilmiş strateji yok — reviewer raporu üretilmedi.</div>'

    reviewer = StatisticalReviewer()
    cards = []
    for hid, title, reviews_json in accepted:
        blocks = []
        if reviews_json:
            try:
                for rep in json.loads(reviews_json):
                    blocks.append(_render_report(rep))
            except (json.JSONDecodeError, TypeError):
                pass
        if hid in stat_by_hid:
            blocks.append(_render_report(reviewer.review(stat_by_hid[hid]).model_dump()))
        cards.append(f'<div class="detail"><div class="dh">'
                     f'<span class="did">{_esc(hid)}</span> {_esc(title)}</div>'
                     + "".join(blocks) + "</div>")
    return "".join(cards)


def _procedural(memory_db: str) -> str:
    """Procedural memory (Doküman 12.3): hangi araştırma hamlesi işe yaradı."""
    from memory.procedural import build_procedural_lessons
    store = MemoryStore(memory_db)
    lessons = build_procedural_lessons(store)
    store.close()
    if not lessons:
        return ('<div class="card desc">Henüz süreç dersi çıkmadı (türetilmiş '
                'hipotez / yeterli deney yok).</div>')
    items = "".join(f'<div class="drow">• {_esc(l)}</div>' for l in lessons)
    return f'<div class="card">{items}</div>'


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
        _section("Tüm Denenen Hipotezler — LLM Ne Denedi, Ne Oldu, Niçin?",
                 "Kampanyanın asıl hikâyesi. LLM'in ürettiği her hipotez düz Türkçe "
                 "olarak ne yaptığıyla birlikte listelenir; yanında sonucu (KABUL / RED "
                 "/ TEKRAR) ve — reddedildiyse — insan diliyle nedeni yazar. Kabul "
                 "çıkmasa bile sistemin neyi neden elediği buradan net görülür.",
                 _all_hypotheses(conn)),
        _section("En İyi Stratejiler",
                 "Tüm süzgeçlerden geçip kabul edilen stratejiler, araştırma dönemi "
                 "Sharpe oranına göre sıralı.",
                 _leaderboard(conn)),
        _section("Hipotez Detayı — Bir Strateji Neyden İbaret?",
                 "Leaderboard'daki kısa başlık yalnızca etikettir. Her hipotez aslında "
                 "şu zengin içeriği taşır: test edilebilir iddia, ekonomik mekanizma, "
                 "beklenen başarısızlık koşulları, asıl DSL sinyal formülü ve sonuçları "
                 "görmeden taahhüt edilen çürütme eşiği (ön kayıt).",
                 _details(conn)),
        _section("Çoklu Test Düzeltmesi — 'Kabul' ≠ 'İstatistiksel Geçerli'",
                 "Çok sayıda deneme yapıldığında yüksek bir Sharpe tesadüfen çıkabilir. "
                 "Deflated Sharpe (DSR) ve FDR bunu düzeltir: FDR 'GEÇTİ' değilse sonuç "
                 "istatistiksel olarak kanıtlanmış sayılmaz.",
                 _multiple_testing(memory_db)),
        _section("Bağımsız Reviewer Ajanları (Doküman 15)",
                 "Üretici LLM'den AYRI, deterministik iki denetçi. Backtest Auditor "
                 "backtest'in GEÇERLİLİĞİNİ denetler (sızıntı/survivorship/maliyet/"
                 "likidite); Statistical Reviewer 'kabul' ile 'istatistiksel doğrulandı'yı "
                 "ayırır (FDR/DSR/güven aralığı/fold). TEMİZ/DİKKAT/SORUN her kontrol için.",
                 _reviewers(memory_db)),
        _section("Çok Amaçlı Sıralama (Pareto)",
                 "Kabul edilen stratejiler tek Sharpe ile değil; Sharpe alt güven "
                 "sınırı, drawdown ve turnover birlikte değerlendirilir. Pareto-optimal "
                 "olanlar hiçbir boyutta başkasına tümüyle yenik düşmez — reward "
                 "hacking'e karşı ek bir süzgeç.",
                 _pareto(memory_db)),
        _section("Kilitli Dönem Sınavı (Holdout)",
                 "Araştırma sırasında hiç görülmeyen, kilitli bir dönemde yapılan son "
                 "test. Bir stratejinin gerçekten genelleyip genellemediği buradan "
                 "anlaşılır (araştırma ajanı bu veriye asla erişemez).",
                 _holdout(holdout_db)),
        _section("Aile Performansı — Bütçe Dağılımı",
                 "Her strateji ailesinin kabul/toplam oranı. Sistem araştırma bütçesini "
                 "başarılı ailelere Thompson sampling (bandit) ile kaydırır.",
                 _families(conn)),
        _section("Hipotez Soy Ağacı (Lineage)",
                 "Bir hipotezin başka bir hipotezden nasıl türetildiği: revision "
                 "(champion'ı geliştir), inversion (başarısızı ters çevir). Araştırmanın "
                 "kör deneme değil, yönlü bir keşif olduğunu gösterir.",
                 _lineage(conn)),
        _section("Süreç Hafızası (Procedural Memory, Doküman 12.3)",
                 "Sistem yalnızca 'hangi faktör iyi'yi değil, 'hangi ARAŞTIRMA "
                 "HAMLESİ işe yarıyor'u da öğrenir: revizyon/ters-çevirme/birleştirme "
                 "kabul oranları, doygun aileler ve en çok elemenin yapıldığı aşama. "
                 "Bu dersler bir sonraki hipotez üretimine geri beslenir.",
                 _procedural(memory_db)),
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
