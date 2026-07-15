# public/documents/generator/demo.py
import os

from reportlab.platypus import SimpleDocTemplate, PageBreak, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_PDF = os.path.normpath(os.path.join(BASE_DIR, "..", "demo.pdf"))
LOGO_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "images", "logo_pdf.png"))
FONT_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "fonts", "Roboto-Regular.ttf"))


def ensure_paths():
    if not os.path.isfile(LOGO_PATH):
        raise FileNotFoundError(f"Missing logo: {LOGO_PATH}")
    if not os.path.isfile(FONT_PATH):
        raise FileNotFoundError(f"Missing font: {FONT_PATH}")


def register_fonts():
    pdfmetrics.registerFont(TTFont("Roboto", FONT_PATH))


def draw_logo_background(canv, doc):
    w, h = A4
    canv.saveState()

    # make it faint if supported
    try:
        if hasattr(canv, "setFillAlpha"):
            canv.setFillAlpha(0.08)
    except Exception:
        pass

    img = ImageReader(LOGO_PATH)
    iw, ih = img.getSize()

    scale = min(w / iw, h / ih) * 0.92
    dw, dh = iw * scale, ih * scale
    x = (w - dw) / 2.0
    y = (h - dh) / 2.0

    canv.drawImage(img, x, y, width=dw, height=dh, mask="auto")
    canv.restoreState()


def generate_demo_pdf():
    ensure_paths()
    register_fonts()

    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    # 3 pages, nothing inside (just tiny spacer so ReportLab is happy)
    story = [
        Spacer(1, 1),
        PageBreak(),
        Spacer(1, 1),
        PageBreak(),
        Spacer(1, 1),
    ]

    doc.build(story, onFirstPage=draw_logo_background, onLaterPages=draw_logo_background)


if __name__ == "__main__":
    generate_demo_pdf()
    print(f"✅ Generated: {OUTPUT_PDF}")
