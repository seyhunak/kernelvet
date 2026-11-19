"""Keep secrets out of logs, reports and child processes."""

from __future__ import annotations

import re

_SECRET_NAME = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|CREDENTIAL|AUTH)",
    re.IGNORECASE,
)

_PATTERNS = [
    # KEY=value / KEY: value where KEY looks secret
    re.compile(
        r"(?P<k>\b[A-Za-z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_?KEY|ACCESS_?KEY)[A-Za-z0-9_]*\b\s*[=:]\s*)"
        r"(?P<v>[^\s'\"]+)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(ghp|gho|ghs|github_pat)_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
]

REDACTED = "[REDACTED]"


def is_secret_name(name: str) -> bool:
    return bool(_SECRET_NAME.search(name))


def redact(text: str | None) -> str:
    if not text:
        return text or ""
    out = text
    for pat in _PATTERNS:
        if "k" in pat.groupindex:
            out = pat.sub(lambda m: m.group("k") + REDACTED, out)
        else:
            out = pat.sub(REDACTED, out)
    return out


def scrub_env(env: dict[str, str]) -> dict[str, str]:
    """Drop secret-looking variables before handing an environment to a child process."""
    return {k: v for k, v in env.items() if not is_secret_name(k)}
