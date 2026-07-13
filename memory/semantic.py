"""
Semantic memory (Doküman 12.2) — deneylerden genellenebilir bilgi çıkarır.

Episodik kayıtları (her deneyin tam kaydı) alıp üreticiye yön verecek kısa
derslere dönüştürür: "Reversal bu veride çalışmıyor", "Momentum umut verici,
geliştir". Bu dersler ResearchContext üzerinden LLM'e geri beslenir; sistem
böylece geçmişten ÖĞRENİR (Deney B: hafızalı vs hafızasız).
"""
from __future__ import annotations

# Umut verici sayılma eşiği (bu veride pozitif ve anlamlı)
PROMISING_SHARPE = 0.3


def build_lessons(family_stats: list[tuple]) -> list[str]:
    """(family, count, avg_sharpe, best_sharpe) -> kısa ders cümleleri."""
    lessons: list[str] = []
    for family, count, avg, best in family_stats:
        avg = avg or 0.0
        best = best or 0.0
        if best >= PROMISING_SHARPE:
            lessons.append(
                f"'{family}' ailesi UMUT VERİCİ (en iyi Sharpe {best:.2f}, {count} deneme) "
                f"— bu yapıyı GELİŞTİR: pencereyi uzat, nötralizasyon ekle ya da "
                f"falsification eşiğini gerçekçi (0.3–0.5) koy.")
        elif avg < 0:
            lessons.append(
                f"'{family}' ailesi bu veride ZAYIF (ort Sharpe {avg:.2f}, {count} deneme) "
                f"— bu mekanizmayı tekrarlama, farklı bir yön dene.")
        else:
            lessons.append(
                f"'{family}' ailesi belirsiz (ort {avg:.2f}, {count} deneme).")
    return lessons
