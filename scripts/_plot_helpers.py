"""Shared plotting constants and helpers.

Imported by both build_report.py and synthetic_test.py.
Do NOT import either of those modules from here.
"""

# Colours for the three density estimators
_AB_COLOR = "#5711da"
_FB_COLOR  = "#d33be0"
_HIST_COLOR = "#7de1ff"

# Region display names and colours (shared between build_report.py and
# figure modules so that no circular imports are needed)
REGION_NAMES: dict[str, str] = {
    "camels":    "CAMELS \u00b7 USA",
    "camelsgb":  "CAMELS-GB \u00b7 Great Britain",
    "hysets":    "HYSETS \u00b7 North America & México",
    "lamah":     "LamaH-CE \u00b7 Central Europe",
    "camelsaus": "CAMELS-AUS \u00b7 Australia",
    "camelsbr":  "CAMELS-BR \u00b7 Brazil",
    "camelscl": "CAMELS-CL \u00b7 Chile",
}

_REGION_COLORS: dict[str, str] = {
    "camelsgb": "#e29b03",
    "hysets":   "#3373eb",
    "lamah":    "#924E00",
    "camelsaus": "#f74c4c",
    "camelsbr": "#34c03b",
    "camelscl": "#59d8ff",    
}

_BODY_FONT       = "EB Garamond, Noto Serif, serif"
_LEGEND_FONT_SIZE = "11px"

def _apply_theme(fig) -> None:
    """Apply the shared report figure theme in-place."""
    fig.background_fill_color = "#fffff8"
    fig.border_fill_color     = "#fffff8"
    fig.grid.grid_line_color  = "#b8b8b8"
    fig.axis.axis_label_text_font        = _BODY_FONT
    fig.axis.axis_label_text_font_style  = "normal"
    fig.axis.major_label_text_font       = _BODY_FONT
    fig.axis.major_label_text_font_style = "normal"
    fig.axis.axis_label_text_font_size   = "14px"
    fig.axis.major_label_text_font_size  = "14px"
    fig.title.text_font                  = _BODY_FONT
    fig.title.text_font_size             = "14px"


def _style_legend(fig, location: str, font_size: str = _LEGEND_FONT_SIZE) -> None:
    """Apply the standard legend style to a figure's legend in-place."""
    fig.legend.location              = location
    fig.legend.label_text_font       = _BODY_FONT
    fig.legend.label_text_font_size  = font_size
    fig.legend.click_policy          = "hide"
    fig.legend.background_fill_alpha = 0.7


# Keyword-argument dicts for Bokeh object constructors: avoids repeating font
# strings at every call site.
_LEGEND_STYLE_KW: dict = dict(
    label_text_font=_BODY_FONT,
    label_text_font_size=_LEGEND_FONT_SIZE,
    click_policy="hide",
    background_fill_alpha=0.7,
)

_SIDE_TITLE_KW: dict = dict(
    text_font=_BODY_FONT,
    text_font_size="14px",
    text_font_style="normal",
    text_color="#444444",
)


def _apply_dotwhisker_axis_style(fig) -> None:
    """Tufte-minimal axis style for dotwhisker CI panels (nested-category y-axis)."""
    fig.grid.grid_line_color         = None
    fig.outline_line_color           = None
    fig.xaxis.axis_line_color        = "#aaaaaa"
    fig.yaxis.axis_line_color        = None
    fig.yaxis.major_tick_line_color  = None
    fig.xaxis.minor_tick_line_color  = None
    fig.yaxis.group_text_font            = _BODY_FONT
    fig.yaxis.group_text_font_size       = "13px"
    fig.yaxis.major_label_text_font      = _BODY_FONT
    fig.yaxis.major_label_text_font_size = "13px"
    fig.xaxis.major_label_text_font      = _BODY_FONT
    fig.xaxis.major_label_text_font_size = "13px"
    fig.xaxis.axis_label_text_font       = _BODY_FONT
    fig.xaxis.axis_label_text_font_size  = "12px"
    fig.xaxis.major_label_orientation    = 0.785
