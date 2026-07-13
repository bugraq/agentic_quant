"""
Static Validator — strateji ÇALIŞTIRILMADAN önce sızıntı ve geçerlilik kontrolü.

En kritik kontrol (Doküman 6.1): sinyalin dayandığı en geç bilgi anı, işlem
anından KESİNLİKLE önce olmalı:

    signal.info_tick  <  execution.trade_time.info_tick

Bunun yanında: yasak zaman yönü (forward), parametre aralıkları (negatif/aşırı
pencere), ve karmaşıklık üst sınırı kontrol edilir. Çıktı yapılandırılmış
Decision'dır (serbest metin değil).
"""
from __future__ import annotations

from contracts.decision import (
    Decision,
    DecisionSource,
    DecisionType,
    Issue,
    Severity,
)
from contracts.hypothesis_spec import HypothesisSpec
from contracts.strategy_graph import GraphNode, StrategyGraph
from dsl.operators import FORWARD, get_operator, parse_time_token, tick_to_label

# Karmaşıklık üst sınırları (Doküman 6.4) — aşırı karmaşık = overfitting riski.
MAX_NODES = 40
MAX_DEPTH = 12
MAX_FREE_PARAMS = 10


def _node_map(graph: StrategyGraph) -> dict[str, GraphNode]:
    return {n.node_id: n for n in graph.nodes}


def _subtree_sig(nid: str, nodes: dict[str, GraphNode]):
    """Bir alt ağacın yapısal imzası (dejenere koşul tespiti için)."""
    n = nodes[nid]
    params = tuple(sorted((k, v) for k, v in n.params.items() if k != "_info_tick"))
    return (n.op, params, tuple(_subtree_sig(c, nodes) for c in n.input_ids))


def _find_degenerate_conditionals(nodes: dict[str, GraphNode]) -> list[str]:
    """İki değer-dalı yapısal olarak AYNI olan conditional'lar = sahte koşullama."""
    bad = []
    for n in nodes.values():
        if n.op == "conditional" and len(n.input_ids) == 3:
            _, a, b = n.input_ids
            if _subtree_sig(a, nodes) == _subtree_sig(b, nodes):
                bad.append(n.node_id)
    return bad


def validate(graph: StrategyGraph, hyp: HypothesisSpec) -> Decision:
    """StrategyGraph + execution bağlamı -> Decision (accept/revise/reject)."""
    issues: list[Issue] = []
    reject_level = False  # yapısal olarak imkansız (düzeltilemez) hata var mı

    nodes = _node_map(graph)

    # --- 1) Yasak zaman yönü + parametre aralıkları (düğüm bazında) ---
    for node in graph.nodes:
        if node.op in ("field", "const"):
            continue
        spec = get_operator(node.op)
        if spec is None:
            issues.append(Issue(type="unknown_operator",
                                description=f"Kayıtta olmayan operatör: {node.op}"))
            reject_level = True
            continue
        if spec.time_direction == FORWARD:
            issues.append(Issue(type="forward_leakage",
                                description=f"{node.op} geleceğe bakıyor (forward)."))
            reject_level = True
        if spec.needs_window:
            w = node.params.get("window")
            if w is None or w < spec.window_min:
                issues.append(Issue(
                    type="invalid_parameter",
                    description=f"{node.op}: geçersiz pencere {w} (min {spec.window_min}).",
                    required_action="Pozitif, makul bir pencere kullan."))
                reject_level = True
            elif w > spec.window_max:
                issues.append(Issue(
                    type="invalid_parameter",
                    description=f"{node.op}: pencere {w} aşırı uzun (max {spec.window_max}).",
                    required_action="Lookback'i kısalt."))
                reject_level = True

    # --- 2) SIZINTI kontrolü: signal.info_tick < trade_time.info_tick ---
    signal_node = nodes.get(graph.signal_node_id)
    leak = False
    if signal_node is None:
        issues.append(Issue(type="internal", description="Signal düğümü bulunamadı."))
        reject_level = True
    else:
        signal_tick = int(signal_node.params.get("_info_tick", 0))
        try:
            trade_tick = parse_time_token(hyp.execution.trade_time)
        except ValueError as e:
            issues.append(Issue(type="invalid_execution", description=str(e)))
            reject_level = True
            trade_tick = None

        if trade_tick is not None and not signal_tick < trade_tick:
            leak = True
            issues.append(Issue(
                type="temporal_leakage",
                description=(
                    f"Sinyal {tick_to_label(signal_tick)} bilgisine dayanıyor, "
                    f"işlem {hyp.execution.trade_time} ({tick_to_label(trade_tick)}) "
                    f"anında; bilgi işlemden önce bilinmiyor."),
                required_action="İşlemi en az bir bar sonraya kaydır (örn. open_t_plus_1)."))

    # --- 2b) Dejenere koşul: iki dalı aynı conditional = sahte koşullama ---
    for nid in _find_degenerate_conditionals(nodes):
        issues.append(Issue(
            type="degenerate_conditional",
            description=("conditional'ın iki değer-dalı aynı — koşul boşa çalışıyor "
                         "(sahte rejim/koşullama yapısı)."),
            required_action="İki dalı anlamlı biçimde farklılaştır ya da conditional'ı kaldır."))
        reject_level = True

    # --- 3) Karmaşıklık üst sınırı ---
    c = graph.complexity
    if c.node_count > MAX_NODES or c.depth > MAX_DEPTH or c.free_parameters > MAX_FREE_PARAMS:
        issues.append(Issue(
            type="excessive_complexity",
            description=(f"Karmaşıklık sınırı aşıldı "
                         f"(nodes={c.node_count}, depth={c.depth}, params={c.free_parameters})."),
            required_action="Stratejiyi sadeleştir."))
        reject_level = True

    # --- Karar birleştirme ---
    if not issues:
        return Decision(hypothesis_id=hyp.hypothesis_id, decision=DecisionType.accept,
                        source=DecisionSource.gate, severity=Severity.low)
    if reject_level:
        return Decision(hypothesis_id=hyp.hypothesis_id, decision=DecisionType.reject,
                        source=DecisionSource.gate, severity=Severity.high, issues=issues)
    # sadece sızıntı gibi düzeltilebilir sorun -> revise
    return Decision(
        hypothesis_id=hyp.hypothesis_id, decision=DecisionType.revise,
        source=DecisionSource.gate,
        severity=Severity.high if leak else Severity.medium,
        issues=issues,
        revision_direction="İşlemi sinyal anından sonraya kaydır." if leak else None)
