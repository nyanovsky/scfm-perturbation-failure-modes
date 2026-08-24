"""HTML pieces for the report page.

Only structure lives here: the theme the Lab injects owns every visual choice,
so this module emits semantic HTML and the house figure wrapper and nothing
else. The prose lives in ``build.py`` next to the numbers it quotes.
"""

from __future__ import annotations

import html
from collections.abc import Iterable


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def section(title: str, body: str, sources: Iterable[str] = (), level: int = 2) -> str:
    src = f' data-sources="{";".join(sources)}"' if sources else ""
    head = f"<h{level}>{esc(title)}</h{level}>" if title else ""
    return f"<section{src}>{head}{body}</section>"


def figure(
    number: int,
    title: str,
    chart_html: str,
    description: str,
    findings: str,
    sources: Iterable[str] = (),
) -> str:
    src = f' data-sources="{";".join(sources)}"' if sources else ""
    return (
        f'<section class="figure"{src}>'
        f'<h3 class="figure-title">Figure {number}: {esc(title)}</h3>'
        f"{chart_html}"
        f'<p class="figure-description">{description}</p>'
        f'<p class="figure-findings">{findings}</p>'
        f"</section>"
    )


def table(
    headers: list[str],
    rows: list[list[str]],
    sources: Iterable[str] = (),
    caption: str = "",
) -> str:
    src = f' data-sources="{";".join(sources)}"' if sources else ""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c if c.startswith('<') else esc(c)}</td>" for c in row) + "</tr>"
        for row in rows
    )
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f"<table{src}>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def spec(groups: list[tuple[str, list[tuple[str, str]]]]) -> str:
    parts = []
    for name, rows in groups:
        items = "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in rows)
        parts.append(f"<h3>{esc(name)}</h3><dl>{items}</dl>")
    return f'<div class="spec">{"".join(parts)}</div>'


def aside(body: str, valence: str = "") -> str:
    cls = f"aside {valence}".strip()
    return f'<aside class="{cls}">{body}</aside>'


def key_finding(body: str) -> str:
    return f'<div class="key-finding">{body}</div>'


def document(title: str, blocks: Iterable[str]) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(title)}</title>\n</head>\n<body>\n"
        f'<div class="report">\n{"".join(blocks)}\n</div>\n</body>\n</html>\n'
    )


def num(x: float, digits: int = 4) -> str:
    """Four significant figures, and never scientific notation for a
    correlation-sized number."""
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "n/a"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) < 1e-3:
        return f"{v:.5f}"
    return f"{v:.{digits}g}"


def sig(x: float, digits: int = 4, signed_: bool = False) -> str:
    """Exactly `digits` significant figures, trailing zeros kept.

    ``num`` uses %g, which drops a trailing zero and so shows 0.026196 as
    0.0262. Where a number is quoted against another one - a difference against
    the difference the comparison can resolve - both are written to the same
    number of significant figures instead.
    """
    import math
    v = float(x)
    if v != v:
        return "n/a"
    dec = 0 if v == 0 else max(0, digits - 1 - math.floor(math.log10(abs(v))))
    return f"{v:{'+' if signed_ else ''}.{dec}f}"


def signed(x: float, digits: int = 3) -> str:
    v = float(x)
    return f"{'+' if v >= 0 else ''}{v:.{digits}f}"
