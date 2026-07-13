"""
Compiler — HypothesisSpec'i deterministik StrategyGraph'a çevirir.

Aynı HypothesisSpec her zaman aynı graph'a derlenir. Ağacı düz düğüm
listesine açar, feature referanslarını çözer, her düğüm için `info_tick`
(en erken bilgi anı) ve karmaşıklık ölçülerini hesaplar.

Yapısal hataları burada yakalar (bilinmeyen operatör/alan, tanımsız feature,
arite uyuşmazlığı). Sızıntı ve parametre kontrolü static_validator'da.
"""
from __future__ import annotations

from contracts.dsl import Expression
from contracts.hypothesis_spec import HypothesisSpec
from contracts.strategy_graph import ComplexityMetrics, GraphNode, StrategyGraph
from dsl.operators import (
    CROSS,
    DATA_FIELDS,
    FIELD_BASE_TICK,
    SCALAR,
    SERIES,
    get_operator,
    tick_to_label,
)

CONST_TICK = -10**9  # sabit: her zaman bilinir, max() içinde etkisiz kalır


class CompileError(Exception):
    """Yapısal derleme hatası — hipotez şema-geçerli ama derlenemez."""


class _Builder:
    def __init__(self) -> None:
        self.nodes: list[GraphNode] = []
        self._counter = 0
        self.fields_used: set[str] = set()
        self.free_params = 0
        self.conditions = 0

    def _new_id(self, op: str) -> str:
        self._counter += 1
        return f"n{self._counter}_{op}"

    def build(
        self,
        expr: Expression,
        feature_ticks: dict[str, int],
        feature_types: dict[str, str],
        feature_ids: dict[str, str],
    ) -> tuple[str, int, str, int]:
        """
        Bir ifadeyi düğümlere açar.
        Döndürür: (node_id, info_tick, output_type, depth)
        """
        # --- Yapraklar ---
        if expr.op == "field":
            if not expr.field or expr.field not in DATA_FIELDS:
                raise CompileError(f"Bilinmeyen/izinsiz veri alanı: {expr.field!r}")
            self.fields_used.add(expr.field)
            tick = FIELD_BASE_TICK[expr.field]
            nid = self._add(expr.op, {"field": expr.field}, [], SERIES, "backward", tick)
            return nid, tick, SERIES, 1

        if expr.op == "const":
            nid = self._add(expr.op, {"value": expr.value}, [], SCALAR, "pointwise", CONST_TICK)
            return nid, CONST_TICK, SCALAR, 1

        if expr.op == "feature_ref":
            if not expr.name or expr.name not in feature_ticks:
                raise CompileError(f"Tanımsız feature referansı: {expr.name!r}")
            # feature_ref'in kendisi bir düğüm değil; hedef feature'ın kimliğini taşır
            return feature_ids[expr.name], feature_ticks[expr.name], feature_types[expr.name], 1

        # --- Operatörler ---
        spec = get_operator(expr.op)
        if spec is None:
            raise CompileError(f"Bilinmeyen operatör: {expr.op!r}")

        n_inputs = len(expr.inputs)
        if not (spec.min_arity <= n_inputs <= spec.max_arity):
            raise CompileError(
                f"{expr.op}: arite hatası (verilen {n_inputs}, "
                f"beklenen {spec.min_arity}..{spec.max_arity})"
            )

        # Alt ifadeleri derle
        child_ids: list[str] = []
        child_ticks: list[int] = []
        child_types: list[str] = []
        max_child_depth = 0
        for inp in expr.inputs:
            sub = inp if isinstance(inp, Expression) else Expression(op="feature_ref", name=inp)
            cid, ctick, ctype, cdepth = self.build(sub, feature_ticks, feature_types, feature_ids)
            child_ids.append(cid)
            child_ticks.append(ctick)
            child_types.append(ctype)
            max_child_depth = max(max_child_depth, cdepth)

        # Parametreler
        params: dict = dict(expr.params)
        window = expr.window
        if spec.needs_window:
            if window is None:
                raise CompileError(f"{expr.op}: window parametresi zorunlu")
            params["window"] = window
            self.free_params += 1
        if expr.op == "conditional":
            self.conditions += 1

        # info_tick hesabı
        base_tick = max(child_ticks) if child_ticks else 0
        if spec.is_lag:
            k = int(window or 0)
            info_tick = base_tick - 2 * k   # lag GERİYE kaydırır (güvenli yön)
        else:
            info_tick = base_tick

        out_type = self._resolve_type(spec.output_type, child_types)
        nid = self._add(expr.op, params, child_ids, out_type, spec.time_direction, info_tick)
        return nid, info_tick, out_type, max_child_depth + 1

    @staticmethod
    def _resolve_type(declared: str, child_types: list[str]) -> str:
        """NUMERIC polimorfik tipi çocuklara göre somutlaştır (CROSS > SERIES > SCALAR)."""
        if declared != "numeric":
            return declared
        if CROSS in child_types:
            return CROSS
        if SERIES in child_types:
            return SERIES
        return SCALAR

    def _add(self, op, params, input_ids, out_type, direction, tick) -> str:
        nid = self._new_id(op)
        node_params = dict(params)
        node_params["_info_tick"] = tick  # validator'ın okuyacağı sayısal tick
        self.nodes.append(
            GraphNode(
                node_id=nid,
                op=op,
                params=node_params,
                input_ids=input_ids,
                output_type=out_type,
                time_direction=direction,
                min_lookback=int(params.get("window", 0) or 0),
                max_info_time=tick_to_label(tick) if tick != CONST_TICK else "const",
            )
        )
        return nid


def compile_hypothesis(hyp: HypothesisSpec) -> StrategyGraph:
    """HypothesisSpec -> StrategyGraph (deterministik)."""
    b = _Builder()
    feature_ticks: dict[str, int] = {}
    feature_types: dict[str, str] = {}
    feature_ids: dict[str, str] = {}

    # Feature'lar sırayla derlenir (sonraki, önceki feature'a atıf yapabilir)
    for feat in hyp.features:
        nid, tick, out_type, _ = b.build(
            feat.expression, feature_ticks, feature_types, feature_ids
        )
        feature_ticks[feat.name] = tick
        feature_types[feat.name] = out_type
        feature_ids[feat.name] = nid

    # Signal derle
    signal_id, _, _, depth = b.build(hyp.signal, feature_ticks, feature_types, feature_ids)

    complexity = ComplexityMetrics(
        node_count=len(b.nodes),
        depth=depth,
        free_parameters=b.free_params,
        conditions=b.conditions,
        data_sources=len(b.fields_used),
    )
    return StrategyGraph(
        hypothesis_id=hyp.hypothesis_id,
        nodes=b.nodes,
        feature_node_ids=feature_ids,
        signal_node_id=signal_id,
        required_data_fields=sorted(b.fields_used),
        complexity=complexity,
    )
