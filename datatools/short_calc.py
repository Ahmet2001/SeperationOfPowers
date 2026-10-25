"""Teacher-free short everyday calculations (PROJECT_MASTER §7).

The CALCULATE sources are long translated word problems (``arac_kullanimi``
tool-math); the end-to-end CLI sent requests such as "1250 lirayı 8 kişiye
bölersek kişi başı ne düşer?" to RESPOND and Yargi then guessed the result.
Each episode here is one templated Turkish request, the expression built from
its own numbers, the runtime CALCULATE observation and a templated answer.
Rows use the source formats, so build_shared_data normalizes and verifies them
like any other source: Yasama gets the CALCULATE step and the FINISH step,
Yurutme the expression, Yargi (a YARGI_SHARE of the episodes) the answer.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from runtime.local_tools import execute_builtin_local

SOURCE = "short_calc"
EPISODES = 10_000
YARGI_SHARE = 0.3  # templated answers; keep them a minority of Yargi's CALCULATE stratum
SEED = "short_calc_v1"


@dataclass(frozen=True)
class Template:
    name: str
    numbers: Callable[[random.Random], dict[str, float]]
    prompts: tuple[str, ...]
    expression: str
    answers: tuple[str, ...]


def _money(rng: random.Random, low: int, high: int) -> int:
    value = rng.randint(low, high)
    return value if value < 1000 or rng.random() < 0.5 else round(value, -1)


def _pick(rng: random.Random, *values: float) -> float:
    return rng.choice(values)


TEMPLATES: tuple[Template, ...] = (
    Template(
        "split", lambda r: {"a": _money(r, 120, 25_000), "n": r.randint(2, 12)},
        ("{a} lirayı {n} kişiye eşit bölersek kişi başı ne düşer?",
         "{n} kişi {a} TL'lik hesabı eşit paylaşırsa herkes ne kadar öder?",
         "Toplam {a} TL'yi {n} kişi arasında eşit böl, kişi başı kaç TL?",
         "{a} TL'lik faturayı {n} arkadaş bölüşeceğiz, kişi başı ne kadar?"),
        "{a}/{n}",
        ("Kişi başı {r} TL düşer.", "Herkes {r} TL öder.", "{a} TL {n} kişiye bölünce kişi başı {r} TL eder."),
    ),
    Template(
        "vat_remove", lambda r: {"p": _money(r, 50, 60_000), "k": _pick(r, 1, 10, 20, 18, 8)},
        ("KDV dahil {p} TL olan ürünün yüzde {k} KDV hariç fiyatı nedir?",
         "Fiyat {p} TL ve buna yüzde {k} KDV dahil. KDV'siz fiyat kaç?",
         "KDV oranı yüzde {k}, KDV dahil tutar {p} TL. KDV hariç tutarı hesaplar mısın?"),
        "{p}/(1+{k}/100)",
        ("KDV hariç fiyat {r} TL.", "KDV'siz tutar {r} TL olur."),
    ),
    Template(
        "vat_add", lambda r: {"p": _money(r, 50, 60_000), "k": _pick(r, 1, 10, 20, 18, 8)},
        ("{p} TL'ye yüzde {k} KDV eklenince fiyat kaç olur?",
         "KDV hariç {p} TL olan hizmetin yüzde {k} KDV ile toplam tutarı ne?"),
        "{p}*(1+{k}/100)",
        ("KDV dahil fiyat {r} TL olur.", "Toplam tutar {r} TL."),
    ),
    Template(
        "percent", lambda r: {"a": _money(r, 40, 90_000), "k": r.randint(1, 95)},
        ("{a} sayısının yüzde {k} kadarı kaçtır?", "{a} TL'nin yüzde {k} kısmı ne eder?",
         "Yüzde {k} oranında {a} TL ne kadar eder?"),
        "{a}*{k}/100",
        ("Sonuç {r}.", "{a} değerinin yüzde {k} kadarı {r} eder."),
    ),
    Template(
        "discount", lambda r: {"p": _money(r, 30, 50_000), "k": _pick(r, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70)},
        ("{p} TL'lik ürüne yüzde {k} indirim yapılırsa yeni fiyat ne olur?",
         "Etiket fiyatı {p} TL, yüzde {k} indirim var. Kaça alırım?",
         "{p} liralık montta yüzde {k} indirim uygulanınca ödeyeceğim tutar ne?"),
        "{p}*(1-{k}/100)",
        ("İndirimli fiyat {r} TL olur.", "Yüzde {k} indirimle {r} TL ödersiniz."),
    ),
    Template(
        "raise", lambda r: {"p": _money(r, 8_000, 150_000), "k": r.randint(3, 60)},
        ("Maaşım {p} TL, yüzde {k} zam gelirse yeni maaşım ne olur?",
         "{p} TL kiraya yüzde {k} artış yapılırsa yeni kira kaç TL olur?"),
        "{p}*(1+{k}/100)",
        ("Yeni tutar {r} TL olur.", "Yüzde {k} artışla {r} TL olur."),
    ),
    Template(
        "multiply", lambda r: {"a": r.randint(12, 9_999), "b": r.randint(3, 999)},
        ("{a} çarpı {b} kaç eder?", "{a} * {b} işlemini hesapla.", "{a} ile {b} sayısını çarpar mısın?"),
        "{a}*{b}",
        ("Sonuç {r}.", "{a} çarpı {b} = {r}."),
    ),
    Template(
        "divide", lambda r: {"a": r.randint(100, 99_999), "b": r.randint(2, 97)},
        ("{a} bölü {b} kaç?", "{a} / {b} sonucu nedir?", "{a} sayısını {b} sayısına böler misin?"),
        "{a}/{b}",
        ("Sonuç {r}.", "{a} bölü {b} = {r}."),
    ),
    Template(
        "add", lambda r: {"a": r.randint(100, 99_999), "b": r.randint(100, 99_999), "c": r.randint(10, 9_999)},
        ("{a} + {b} + {c} toplamı kaç?", "{a}, {b} ve {c} sayılarını topla.",
         "Bu ay {a}, {b} ve {c} TL harcadım. Toplam ne kadar etti?"),
        "{a}+{b}+{c}",
        ("Toplam {r}.", "Toplam {r} eder."),
    ),
    Template(
        "average", lambda r: {"a": r.randint(30, 100), "b": r.randint(30, 100), "c": r.randint(30, 100)},
        ("{a}, {b} ve {c} sayılarının ortalaması kaç?", "Sınav notlarım {a}, {b} ve {c}. Ortalamam kaç?"),
        "({a}+{b}+{c})/3",
        ("Ortalama {r}.", "Ortalama {r} eder."),
    ),
    Template(
        "installment", lambda r: {"p": _money(r, 1_500, 120_000), "n": _pick(r, 2, 3, 4, 6, 9, 10, 12, 18, 24)},
        ("{p} TL'lik telefonu {n} eşit taksitle alırsam aylık taksit ne kadar olur?",
         "{p} TL'yi {n} ayda eşit ödeyeceğim, aylık kaç TL?"),
        "{p}/{n}",
        ("Aylık taksit {r} TL olur.", "Ayda {r} TL ödersiniz."),
    ),
    Template(
        "travel_time", lambda r: {"d": r.randint(20, 1_500), "v": _pick(r, 40, 50, 60, 70, 80, 90, 100, 110, 120)},
        ("Saatte {v} km hızla giden araç {d} km yolu kaç saatte alır?",
         "{d} km'lik yolu saatte ortalama {v} km ile kaç saatte giderim?"),
        "{d}/{v}",
        ("Yol {r} saat sürer.", "Yaklaşık {r} saatte varırsınız."),
    ),
    Template(
        "fuel", lambda r: {"l": _pick(r, 4, 5, 6, 7, 8, 9, 11), "d": r.randint(30, 2_000)},
        ("100 km'de {l} litre yakan araç {d} km yolda kaç litre yakar?",
         "Arabam 100 km'de {l} litre yakıyor, {d} km için ne kadar yakıt gerekir?"),
        "{l}*{d}/100",
        ("{r} litre yakıt harcar.", "Yaklaşık {r} litre yakıt gerekir."),
    ),
    Template(
        "interest", lambda r: {"p": _money(r, 1_000, 500_000), "k": r.randint(5, 60), "y": r.randint(1, 5)},
        ("{p} TL'ye yıllık yüzde {k} basit faizle {y} yılda ne kadar faiz işler?",),
        "{p}*{k}/100*{y}",
        ("{y} yılda {r} TL faiz işler.", "Basit faiz tutarı {r} TL."),
    ),
    Template(
        "currency", lambda r: {"a": r.randint(5, 20_000), "k": r.choice((32.5, 34.2, 38.75, 41.1, 36.9, 44.3))},
        ("Dolar {k} TL olsa {a} dolar kaç TL eder?", "Kur {k} ise {a} euro kaç lira yapar?"),
        "{a}*{k}",
        ("{r} TL eder.", "Bu kurla {r} TL yapar."),
    ),
    Template(
        "age", lambda r: {"y1": r.randint(1940, 2015), "y2": r.randint(2016, 2040)},
        ("{y1} yılında doğan biri {y2} yılında kaç yaşında olur?",
         "{y1} doğumluyum, {y2} yılında kaç yaşında olacağım?"),
        "{y2}-{y1}",
        ("{r} yaşında olur.", "{y2} yılında {r} yaşında olursunuz."),
    ),
)

PREFIXES = ("", "", "", "Rica etsem ", "Hızlıca söyler misin: ", "Bir hesap yapar mısın? ")


def _prompt_number(key: str, value: float, rng: random.Random) -> str:
    """How a person writes it: thousands with dots sometimes (never in years,
    keys y*), decimal comma."""
    if isinstance(value, float) and not value.is_integer():
        return f"{value:g}".replace(".", ",")
    value = int(value)
    if value >= 1000 and not key.startswith("y") and rng.random() < 0.4:
        return f"{value:,}".replace(",", ".")
    return str(value)


def answer_number(result: float) -> str:
    """Two decimals at most, decimal comma, no thousands separator (an answer
    number must read back as a rounding of the result, §6)."""
    rounded = round(float(result), 2)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").replace(".", ",")


def _expression_number(value: float) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def episodes(count: int = EPISODES) -> Iterator[dict[str, Any]]:
    """Deterministic episodes; ``source_id`` is stable across builds."""
    for index in range(count):
        rng = random.Random(f"{SEED}:{index}")
        template = TEMPLATES[index % len(TEMPLATES)]
        numbers = template.numbers(rng)
        shown = {k: _prompt_number(k, v, rng) for k, v in numbers.items()}
        prefix, body = rng.choice(PREFIXES), rng.choice(template.prompts).format(**shown)
        if prefix.endswith(" ") and not prefix.endswith(("? ", ": ")) and body[1:2].islower():
            body = body[0].lower() + body[1:]  # "Rica etsem arabam ..." but "KDV"
        prompt = prefix + body
        expression = template.expression.format(**{k: _expression_number(v) for k, v in numbers.items()})
        observation = execute_builtin_local("CALCULATE", {"expression": expression})
        if observation["status"] != "success":
            continue
        result = observation["data"]["result"]
        answer = rng.choice(template.answers).format(r=answer_number(result), **shown)
        yield {
            "source_id": f"{SEED}-{template.name}-{index}",
            "user_prompt": prompt,
            "expression": expression,
            "observation": observation,
            "result": result,
            "response": answer,
        }


def _share(source_id: str) -> float:
    return int(hashlib.md5(f"{SEED}:yargi:{source_id}".encode()).hexdigest(), 16) / 16**32


def rows(role: str, items: list[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """Source-format rows of one role (see the arac_kullanimi rows)."""
    empty = {"completed_actions": [], "last_observation": None, "recent_observations": []}
    for item in items:
        base = {"source_id": item["source_id"], "turn_index": 0, "user_prompt": item["user_prompt"],
                "conversation_history": []}
        observation = item["observation"]
        if role == "yasama":
            yield {**base, "step_index": 0, "state": empty, "target_action": "CALCULATE"}
            done = {"completed_actions": ["CALCULATE"], "last_observation": observation,
                    "recent_observations": [observation]}
            yield {**base, "step_index": 1, "state": done, "target_action": "FINISH"}
        elif role == "yurutme":
            yield {**base, "step_index": 0, "state": empty, "mapped_action": "CALCULATE",
                   "action": "CALCULATE", "target_arguments": {"expression": item["expression"]}}
        elif _share(item["source_id"]) < YARGI_SHARE:
            yield {**base, "observations": [observation], "response": item["response"]}
