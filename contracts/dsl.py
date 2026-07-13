"""
DSL ifade ağacı — LLM'in üretebileceği tek strateji dili.

Bir Expression, operatörlerden oluşan bir ağaçtır. Yaprakları ham veri
alanları (field), adlandırılmış feature referansları (feature_ref) veya
sabitler (const) olabilir. İç düğümler onaylı operatörlerdir.

Bu modül SADECE ifadenin YAPISINI tanımlar. Operatörün ne yaptığı,
zaman yönü, tip bilgisi gibi anlamsal metadata `dsl/operators` içinde;
sızıntı kontrolü ise `dsl/static_validator` içinde yaşayacak.
"""
from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field

# Bir input ya iç içe bir ifade ya da bir feature adı (string kısayolu) olabilir.
# Doküman 4.4'teki örnekte inputs hem obje hem string içeriyordu; ikisini de destekliyoruz.
InputType = Union["Expression", str]


class Expression(BaseModel):
    """
    Tek bir DSL düğümü.

    op alanı düğümün türünü söyler:
      - "field"        → ham veri alanı (field="close")
      - "feature_ref"  → başka bir feature'a atıf (name="residual_return_20d")
      - "const"        → sabit sayı (value=0.1)
      - <operator adı> → onaylı operatör (op="rolling_mean", window=20, inputs=[...])
    """

    op: str = Field(..., description="Operatör adı veya 'field'/'feature_ref'/'const'")

    # Yaprak alanları (op'a göre biri dolar)
    field: Optional[str] = Field(None, description="op='field' için ham veri alanı adı")
    name: Optional[str] = Field(None, description="op='feature_ref' için feature adı")
    value: Optional[float] = Field(None, description="op='const' için sabit değer")

    # Operatör parametreleri
    window: Optional[int] = Field(None, description="Pencere uzunluğu (varsa)")
    params: dict = Field(default_factory=dict, description="Ek operatör parametreleri")

    # Alt ifadeler (iç düğümler için)
    inputs: list[InputType] = Field(default_factory=list)

    model_config = {"extra": "forbid"}  # bilinmeyen alan = hata (sızıntıya kapı açmayalım)

    def is_leaf(self) -> bool:
        return self.op in ("field", "feature_ref", "const")


class NamedFeature(BaseModel):
    """HypothesisSpec içindeki adlandırılmış feature: name + ifade."""

    name: str
    expression: Expression


# Pydantic'in recursive tip çözümlemesi için:
Expression.model_rebuild()
