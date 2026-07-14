"""
Procedural memory (Doküman 12.3 + AlphaMemo literatürü).

Semantic memory "hangi FAKTÖR ailesi iyi/kötü" der; procedural memory "hangi
ARAŞTIRMA HAMLESİ (operasyon) işe yarıyor" der. Deney soy ağacından (lineage)
ve aşama sayımından çıkarılır:

  - Operasyon verimliliği: revizyon / ters-çevirme / birleştirme / parametre
    varyantı hamlelerinin kabul oranı. LLM'e "bu kampanyada ters çevirme işe
    yaramadı, birleştirme verimli" gibi geri besleme.
  - Aile doygunluğu: çok denenip az kabul alan aile → doygun/zayıf.
  - Eleme örüntüsü: en çok hangi aşamada eleniyor (gate mi, robustness mı,
    duplicate mı) → LLM neye dikkat etmeli.

Bu bilgi hem generator prompt'una (yön verir) hem dashboard'a (şeffaflık) gider.
AlphaMemo'nun "edit-motif veto"suna doğru ilk adım: yüksek-güvenle verimsiz
hamleleri LLM'e bildirmek.
"""
from __future__ import annotations

_OP_TR = {
    "refinement": "revizyon (champion geliştirme)",
    "inversion": "ters çevirme (başarısızın tersi)",
    "combination": "birleştirme (iki sinyali sentezleme)",
    "parameter_variant": "parametre varyantı (optimizer)",
}

_STAGE_TR = {
    "compile_error": "derleme",
    "static_rejected": "sızıntı/statik",
    "critic_rejected": "critic (ekonomik)",
    "duplicate": "tekrar (novelty)",
    "degenerate_conditional": "ölü koşul",
    "gate_rejected": "performans kapısı",
    "robustness_rejected": "sağlamlık",
}
_REJECT_STAGES = {"compile_error", "static_rejected", "critic_rejected",
                  "gate_rejected", "robustness_rejected", "degenerate_conditional"}


def _operation_lessons(edges: list[tuple]) -> list[str]:
    """lineage kenarlarından (parent, child, relation_type, decision) operasyon verimi."""
    by_op: dict[str, list[int]] = {}
    for _parent, _child, rel, dec in edges:
        # parametre varyantı optimizer'ın işidir; LLM hamlesi değil — atla.
        if not rel or rel == "parameter_variant":
            continue
        a, t = by_op.get(rel, (0, 0))
        by_op[rel] = (a + (1 if dec == "accept" else 0), t + 1)
    lessons = []
    for rel, (a, t) in sorted(by_op.items(), key=lambda x: -x[1][1]):
        if t < 2:
            continue
        rate = a / t
        verdict = ("verimli — bu hamleyi sürdür" if rate >= 0.3 else
                   ("düşük verim — dikkatli kullan" if a > 0 else
                    "bu kampanyada hiç kabul getirmedi — farklı hamle dene"))
        lessons.append(f"{_OP_TR.get(rel, rel)}: {t} denemenin {a}'ı kabul ({verdict}).")
    return lessons


def _saturation_lessons(family_stats: list[tuple]) -> list[str]:
    """family_stats: (family, count, avg_sharpe, best_sharpe) — doygun/zayıf aileler."""
    lessons = []
    for fam, count, avg_sh, best_sh in family_stats:
        if count and count >= 4 and (best_sh is None or best_sh < 0.3):
            lessons.append(f"'{fam}' ailesi {count} kez denendi, en iyi Sharpe "
                           f"{(best_sh or 0):.2f} — doygun/zayıf, yeni yön ara.")
    return lessons


def _stage_lessons(stage_counts: dict[str, int]) -> list[str]:
    """En çok elemenin yapıldığı aşamayı öne çıkar."""
    rej = {k: v for k, v in stage_counts.items() if k in _REJECT_STAGES and v}
    if not rej:
        return []
    top = max(rej, key=rej.get)
    total_rej = sum(rej.values())
    return [f"Elemelerin çoğu '{_STAGE_TR.get(top, top)}' aşamasında "
            f"({rej[top]}/{total_rej}) — üretirken buna özellikle dikkat et."]


def build_procedural_lessons(memory) -> list[str]:
    """MemoryStore'dan procedural dersleri derle (Doküman 12.3)."""
    lessons: list[str] = []
    lessons += _operation_lessons(memory.lineage_edges())
    lessons += _saturation_lessons(memory.family_stats())
    lessons += _stage_lessons(memory.stage_counts())
    return lessons
