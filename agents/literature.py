"""
Literatür ajanı (Doküman 4.3 — 'literatürdeki mekanizmayı yeni alana uygula').

LLM'e web araması (OpenRouter web_search tool) yaptırıp KANITLANMIŞ kesitsel
hisse getirisi faktörlerini/anomalilerini çeker. Bu mekanizmalar hipotez
üreticiye bağlam olarak verilir; böylece fikirler rastgele değil, gerçek
literatüre dayalı ve çeşitli olur.

Maliyet: web araması normal çağrıdan pahalı. Bu yüzden kampanya başına BİR KEZ
çağrılır ve sonuç tüm iterasyonlarda yeniden kullanılır.
"""
from __future__ import annotations

from llm.openai_client import OpenAICompatibleClient

_SYSTEM = ("Sen bir kantitatif finans literatürü araştırmacısısın. Web'de arama "
           "yaparak akademik olarak BELGELENMİŞ, kesitsel (cross-sectional) hisse "
           "senedi getirisi faktörlerini/anomalilerini bulursun.")


def fetch_literature_mechanisms(client: OpenAICompatibleClient, model: str,
                                universe_description: str, n: int = 6) -> list[str]:
    """Web araması ile n adet gerçek faktör/mekanizma (tek satırlık) döndürür."""
    user = (f"Evren: {universe_description}\n\n"
            f"Web'de ara ve kesitsel hisse getirisini öngördüğü akademik olarak "
            f"belgelenmiş {n} faktör/anomali bul. Yalnızca fiyat ve hacim verisiyle "
            f"(close, open, high, low, volume, dollar_volume, market_cap) "
            f"hesaplanabilecek olanlara öncelik ver.\n"
            f"Her biri için TEK satır yaz: 'Faktör adı — kısa mekanizma "
            f"(kullanılabilir alanlar: ...)'. Sadece liste, başka açıklama yok.")
    try:
        resp = client.chat(model, _SYSTEM, user, temperature=0.3,
                           force_json=False, max_tokens=800, web_search=True)
    except Exception as e:  # noqa: BLE001 — arama başarısızsa literatürsüz devam
        print(f"[literatür] web araması başarısız ({type(e).__name__}); literatürsüz devam.")
        return []
    lines = []
    for raw in resp.text.splitlines():
        s = raw.strip(" -*•\t")
        # baştaki "1." "2)" gibi numaralandırmayı temizle
        while s[:1].isdigit() or s[:1] in ".)":
            s = s[1:].strip()
        if len(s) > 15:
            lines.append(s)
    return lines[:n]
