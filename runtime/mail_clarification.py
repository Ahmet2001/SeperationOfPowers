"""Deterministic clarification policy for incomplete email requests.

The model chooses actions, but a tool must not receive invented SEND_MAIL
arguments when the user has not provided a recipient, topic or body.
The deliberately conservative checks here target zero-config chat testing;
they do not claim to understand arbitrary natural-language email requests.
"""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I)
_MAIL = re.compile(r"\b(?:mail|e[\s-]?posta)", re.I)
_SEND = re.compile(r"(?:gönder|gonder|yolla|yollamak|atmak|atacağ|atacag)", re.I)
_MAIL_CANCEL = re.compile(
    r"^\s*(?:iptal|vazgeçtim|vazgectim|boş\s*ver|bos\s*ver)[.!?\s]*$"
    r"|\b(?:mail|e[\s-]?posta)\w*[\s\S]{0,80}?\b(?:iptal|vazgeçtim|vazgectim)\b"
    r"|\b(?:mail|e[\s-]?posta)\w*\s+(?:gönderme|gonderme|yollama|atma)\b"
    r"|\b(?:mail|e[\s-]?posta)\w*[\s\S]{0,100}?"
    r"(?:göndermek|gondermek|yollamak|atmak|göndermeyi|gondermeyi|yollamayı)"
    r"\s+istemiyorum\b"
    r"|\b(?:mail|e[\s-]?posta)\w*[\s\S]{0,100}?\b(?:istemiyorum|istemiyoruz)\b",
    re.I,
)

_SUBJECT = (
    re.compile(r"(?im)^\s*(?:konu|başlık)\s*[:=]\s*\S+"),
    re.compile(r"""["“'][^"”']+["”']\s*(?:başlıklı|konulu)""", re.I),
)
_BODY = (
    re.compile(r"(?im)^\s*(?:mesaj|içerik|icerik|gövde)\s*[:=]\s*\S+"),
    re.compile(r"""["“'][^"”']+["”']\s*(?:içerikli|icerikli|mesajlı)""", re.I),
    re.compile(r"(?:kısa|samimi|resmi|selamlama|teşekkür|kutlama|özür)\s+(?:\S+\s+){0,5}(?:mail|e[\s-]?posta)", re.I),
)


def mentions_mail(text: str) -> bool:
    """Whether the text actually refers to email."""
    return bool(_MAIL.search(text))


def is_mail_cancel_request(text: str) -> bool:
    """True for an explicit cancellation or a negated mail-send request."""
    return bool(_MAIL_CANCEL.search(text))


def is_mail_send_request(text: str) -> bool:
    """True only when a mail send is requested rather than declined."""
    return bool(
        _MAIL.search(text)
        and _SEND.search(text)
        and not is_mail_cancel_request(text)
    )


_SMALL_TALK = re.compile(
    r"^\s*(?:selam|merhaba|nasılsın|nasilsin|naber|iyi\s+misin|"
    r"günaydın|gunaydin|iyi\s+akşamlar|iyi\s+aksamlar)"
    r"[\s!?.,]*$",
    re.I,
)


def is_simple_chat_request(text: str) -> bool:
    """Recognize unambiguous greetings for conservative action routing."""
    return bool(_SMALL_TALK.fullmatch(text))


def missing_mail_details(text: str) -> tuple[str, ...]:
    """Required fields inferred only from explicit user wording.

    This is a conservative preflight policy; Yurutme still creates the JSON.
    A body instruction like 'kısa bir selamlama maili' counts as a body brief.
    """
    missing: list[str] = []
    if not _EMAIL.search(text):
        missing.append("to")
    if not any(pattern.search(text) for pattern in _SUBJECT):
        missing.append("subject")
    if not any(pattern.search(text) for pattern in _BODY):
        missing.append("body")
    return tuple(missing)


def clarification_question(missing: tuple[str, ...]) -> str:
    """Ask one missing field at a time; ask the topic first."""
    if "subject" in missing:
        return "Hangi konuda mail yollamak istiyorsunuz?"
    if "to" in missing:
        return "Maili kime göndermek istiyorsunuz? E-posta adresini yazar mısınız?"
    if "body" in missing:
        return "Mailin içeriğinde ne yazmamı istersiniz?"
    return "İsteğinizi gerçekleştirebilmem için biraz daha ayrıntı verebilir misiniz?"


def label_mail_clarification_answer(answer: str, previous_question: str) -> str:
    """Make short chat answers unambiguous for the next role-model call."""
    # The user might supply explicit labels for multiple fields at once.
    if re.search(r"(?im)^\s*(?:konu|başlık|kime|alıcı|mesaj|içerik|gövde)\s*:", answer):
        return answer
    if previous_question.startswith("Hangi konuda mail"):
        return f"Konu: {answer}"
    if previous_question.startswith("Maili kime"):
        return f"Kime: {answer}"
    if previous_question.startswith("Mailin içeriğinde"):
        return f"Mesaj: {answer}"
    return answer
