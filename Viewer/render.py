from __future__ import annotations

from html import escape
from typing import Iterable, Optional

from framework.grids import Grid


# ARC color palette (same mapping as the tester app you referenced).
_SYMBOL_TO_COLOR = {
    0: "#000000",  # black
    1: "#0074D9",  # blue
    2: "#FF4136",  # red
    3: "#2ECC40",  # green
    4: "#FFDC00",  # yellow
    5: "#AAAAAA",  # grey
    6: "#F012BE",  # fuschia
    7: "#FF851B",  # orange
    8: "#7FDBFF",  # teal
    9: "#870C25",  # brown
}


def _cell_size_px(grid: Grid, max_px: int = 26) -> int:
    h = max(1, len(grid))
    w = max(1, len(grid[0]))
    # Aim for a reasonable display size; Streamlit columns handle resizing.
    # Clamp to keep large grids readable.
    size = int(max(6, min(max_px, 360 / max(h, w))))
    return size


def grid_height_px(grid: Grid, *, max_px: int = 26) -> int:
    """Rough height for displaying a grid in Streamlit (for iframe)."""
    h = len(grid) or 1
    size = _cell_size_px(grid, max_px=max_px)
    # Add some margin for title and padding.
    return int(h * size + 80)


def grid_html(
    grid: Grid,
    *,
    show_numbers: bool = False,
    title: str | None = None,
    max_px: int = 26,
    border_color: Optional[str] = None,
    border_width_px: int = 1,
) -> str:
    """Return an HTML snippet rendering a single ARC grid."""
    size = _cell_size_px(grid, max_px=max_px)
    h = len(grid)
    w = len(grid[0]) if h else 0

    title_html = ""
    if title is not None:
        title_html = f"<div class='arc-grid-title'>{escape(title)}</div>"

    rows: list[str] = []
    for r in range(h):
        cells: list[str] = []
        for c in range(w):
            sym = int(grid[r][c])
            color = _SYMBOL_TO_COLOR.get(sym, "#000000")
            text = str(sym) if show_numbers else ""
            cells.append(
                "<div class='arc-cell' "
                f"style='background:{color};width:{size}px;height:{size}px'>"
                f"{escape(text)}</div>"
            )
        rows.append("<div class='arc-row'>" + "".join(cells) + "</div>")

    # Streamlit renders `components.html` inside an iframe, so we include the
    # CSS with every snippet to ensure styling is applied.
    style = """
<style>
.arc-grid-wrap { margin: 0.25rem 0 0.75rem 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.arc-grid-title { font-weight: 600; font-size: 0.9rem; margin-bottom: 0.25rem; opacity: 0.85; }
.arc-grid { display: inline-block; border: 1px solid rgba(120,120,120,0.6); background: rgba(0,0,0,0.2); }
.arc-row { display: flex; }
.arc-cell {
  box-sizing: border-box;
  border-right: 1px solid rgba(85,85,85,0.75);
  border-bottom: 1px solid rgba(85,85,85,0.75);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.7rem;
  line-height: 1;
  color: rgba(255,255,255,0.9);
  user-select: none;
}
.arc-row:last-child .arc-cell { border-bottom: 0; }
.arc-cell:last-child { border-right: 0; }
</style>
"""

    grid_border_style = ""
    if border_color is not None:
        grid_border_style = (
            f" style='border:{border_width_px}px solid {escape(border_color)};'"
        )

    return (
        style
        + "<div class='arc-grid-wrap'>"
        + f"{title_html}"
        + f"<div class='arc-grid'{grid_border_style}>"
        + "".join(rows)
        + "</div></div>"
    )

