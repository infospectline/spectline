# public/documents/generator/complaint.py
import os
from datetime import date
from functools import partial

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_PDF_SK = os.path.normpath(os.path.join(BASE_DIR, "..", "complaint_sk.pdf"))
OUTPUT_PDF_EN = os.path.normpath(os.path.join(BASE_DIR, "..", "complaint_en.pdf"))

LOGO_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "..", "..", "public", "images", "logo_pdf.png")
)
FONT_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "..", "..", "public", "fonts", "Roboto-Regular.ttf")
)

ACCENT = colors.HexColor("#1A1A1A")
ACCENT_DARK = colors.HexColor("#1A1A1A")
INK = colors.HexColor("#0B0B0C")
MUTED = colors.HexColor("#5B6472")
LINE = colors.HexColor("#D9E2F2")

TOTAL_WIDTH = A4[0] - 4 * cm
HEADER_ROW_HEIGHT = 26


DATA = {
    "complaint_id": date.today().strftime("%d%m%Y"),

    "provider": {
        "name": "Spectline",
        "country": "Slovensko",
        "ico": "",
        "dic": "",
        "ic_dph": "",
        "registration": "",
        "support_email": "",
        "support_phone": "",
    },

    "customer": {
        "name": "",
        "country": "",
        "address": "",
        "email": "",
        "phone": "",
        "ico": "",
        "dic": "",
    },

    "references": {
        "plan": "",
        "billing_period": "",
        "invoice_id": "",
        "order_id": "",
        "transaction_id": "",
        "account_email": "",
    },

    "dates": {
        "purchase": "",
        "activation": "",
        "claim": date.today().strftime("%d.%m.%Y"),
        "incident": "",
    },

    "items": [
        {
            "service": "",
            "price": "",
            "issue": "",
            "impact": "",
        }
    ],

    "problem": {
        "summary": "",
        "description": "",
        "repro_steps": "",
        "errors": "",
        "environment": "",
    },

    "attachments": [""],
    "requested_resolution": "",
    "currency": "EUR",
}


LANGUAGES = {
    "sk": {
        "complaint_title": "REKLAMÁCIA",
        "complaint_number": "Číslo reklamácie",
        "provider_title": "POSKYTOVATEĽ",
        "customer_title": "KLIENT",
        "company": "SPOLOČNOSŤ",
        "name": "MENO",
        "address": "ADRESA",
        "email": "E-MAIL",
        "phone": "TEL.",
        "company_id": "IČO",
        "tax_id": "DIČ",
        "vat_id": "IČ DPH",
        "purchase_date": "DÁTUM NÁKUPU",
        "delivery_date": "DÁTUM DODANIA",
        "problem_date": "DÁTUM PROBLÉMU",
        "claim": "REKLAMÁCIA",
        "services_section": "Reklamované služby / časti systému:",
        "service_type": "Typ služby",
        "price": "Cena",
        "problem": "Problém",
        "summary": "Stručné zhrnutie",
        "description": "Popis problému",
        "statement": "Týmto uplatňujem reklamáciu vizuálneho obsahu vytvoreného štúdiom Spectline.",
        "signature_note": "Meno, dátum, podpis",
    },
    "en": {
        "complaint_title": "COMPLAINT",
        "complaint_number": "Complaint number",
        "provider_title": "PROVIDER",
        "customer_title": "CLIENT",
        "company": "COMPANY",
        "name": "NAME",
        "address": "ADDRESS",
        "email": "E-MAIL",
        "phone": "TEL.",
        "company_id": "COMPANY ID",
        "tax_id": "TAX ID",
        "vat_id": "VAT ID",
        "purchase_date": "PURCHASE DATE",
        "delivery_date": "DELIVERY DATE",
        "problem_date": "PROBLEM DATE",
        "claim": "COMPLAINT",
        "services_section": "Claimed services / parts of the system:",
        "service_type": "Service type",
        "price": "Price",
        "problem": "Problem",
        "summary": "Brief summary",
        "description": "Problem description",
        "statement": "I hereby submit a complaint regarding visual content created by Spectline studio.",
        "signature_note": "Name, date, signature",
    },
}


def ensure_paths() -> None:
    if not os.path.isfile(FONT_PATH):
        raise FileNotFoundError(f"Missing font: {FONT_PATH}")
    if not os.path.isfile(LOGO_PATH):
        raise FileNotFoundError(f"Missing logo: {LOGO_PATH}")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Roboto", FONT_PATH))


def draw_watermark_fullpage(canv, doc, title: str) -> None:
    w, h = A4

    canv.saveState()
    try:
        if hasattr(canv, "setFillAlpha"):
            canv.setFillAlpha(0.06)
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

    canv.saveState()
    canv.setFont("Roboto", 28)
    canv.setFillColor(ACCENT)
    canv.drawRightString(w - 2 * cm, h - 2.2 * cm, title)
    canv.restoreState()


def p(text: str, style: ParagraphStyle) -> Paragraph:
    safe = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return Paragraph(safe, style)


def box(
    title: str,
    lines: list[str],
    styles: dict,
    width: float,
    body_height: float | None = None,
) -> Table:
    head = Table(
        [[p(title, styles["BOX_HEAD"])]],
        colWidths=[width],
        rowHeights=[16],
        splitByRow=0,
    )
    head.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.Color(1, 1, 1, alpha=0)),
                ("BOX", (0, 0), (-1, -1), 0.8, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )

    body = [p(line, styles["TEXT"]) for line in lines]

    if body_height is not None and len(body) == 1:
        body_tbl = Table(
            [[body[0]]],
            colWidths=[width],
            rowHeights=[body_height],
            splitByRow=0,
        )
    else:
        body_tbl = Table([[line] for line in body], colWidths=[width], splitByRow=0)

    body_tbl.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    outer = Table([[head], [body_tbl]], colWidths=[width], splitByRow=0, hAlign="LEFT")
    outer.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("NOSPLIT", (0, 0), (-1, -1)),
            ]
        )
    )
    return outer


def create_styles() -> dict:
    return {
        "TEXT": ParagraphStyle(
            "TEXT",
            fontName="Roboto",
            fontSize=9.8,
            leading=13,
            textColor=INK,
        ),
        "SMALL": ParagraphStyle(
            "SMALL",
            fontName="Roboto",
            fontSize=9,
            leading=12,
            textColor=MUTED,
        ),
        "BOX_HEAD": ParagraphStyle(
            "BOX_HEAD",
            fontName="Roboto",
            fontSize=9,
            leading=12,
            textColor=ACCENT_DARK,
            spaceAfter=0,
        ),
        "WHITE_HEAD": ParagraphStyle(
            "WHITE_HEAD",
            fontName="Roboto",
            fontSize=9,
            leading=12,
            textColor=colors.white,
            spaceBefore=0,
            spaceAfter=0,
        ),
        "TABLE_CELL": ParagraphStyle(
            "TABLE_CELL",
            fontName="Roboto",
            fontSize=9,
            leading=12,
            textColor=INK,
            spaceBefore=0,
            spaceAfter=0,
        ),
    }


def format_price(raw_price, language: str) -> str:
    if raw_price in ("", None):
        return ""

    try:
        value = float(raw_price)
    except (TypeError, ValueError):
        return str(raw_price)

    if abs(value) < 1e-9:
        return ""

    if language == "sk":
        return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", " ")

    return f"{value:,.2f}"


def generate_pdf(language: str, output_pdf: str) -> None:
    labels = LANGUAGES[language]
    styles = create_styles()

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.7 * cm,
        bottomMargin=2 * cm,
    )

    story = [Spacer(1, 6)]

    cid_tbl = Table(
        [[p(f"{labels['complaint_number']}: {DATA.get('complaint_id', '—')}", styles["SMALL"])]],
        colWidths=[TOTAL_WIDTH],
        hAlign="LEFT",
    )
    cid_tbl.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ]
        )
    )

    story.append(cid_tbl)
    story.append(Spacer(1, 8))

    provider = DATA["provider"]
    customer = DATA["customer"]
    half_width = TOTAL_WIDTH / 2.0

    left_box = box(
        labels["provider_title"],
        [
            f"{labels['company']}: {provider.get('name', '—')}",
            f"{labels['company_id']}: {provider.get('ico', '—')}",
            f"{labels['tax_id']}: {provider.get('dic', '—')}",
            f"{labels['vat_id']}: {provider.get('ic_dph', '—')}",
            f"{labels['email']}: {provider.get('support_email', '—')}",
            f"{labels['phone']}: {provider.get('support_phone', '—')}",
        ],
        styles,
        width=half_width,
    )

    right_box = box(
        labels["customer_title"],
        [
            f"{labels['name']}: {customer.get('name', '—')}",
            f"{labels['address']}: {customer.get('address', '—')}",
            f"{labels['email']}: {customer.get('email', '—')}",
            f"{labels['phone']}: {customer.get('phone', '—')}",
            f"{labels['company_id']}: {customer.get('ico', '—')}",
            f"{labels['tax_id']}: {customer.get('dic', '—')}",
        ],
        styles,
        width=half_width,
    )

    head_tbl = Table(
        [[left_box, right_box]],
        colWidths=[half_width, half_width],
        hAlign="LEFT",
    )
    head_tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    story.append(head_tbl)
    story.append(Spacer(1, 10))

    dates = DATA["dates"]
    dates_tbl = Table(
        [
            [
                p(f"{labels['purchase_date']}:  {dates.get('purchase', '—')}", styles["WHITE_HEAD"]),
                p(f"{labels['delivery_date']}:  {dates.get('activation', '—')}", styles["WHITE_HEAD"]),
                p(f"{labels['problem_date']}:  {dates.get('incident', '—')}", styles["WHITE_HEAD"]),
                p(labels["claim"], styles["WHITE_HEAD"]),
            ]
        ],
        colWidths=[TOTAL_WIDTH / 4.0] * 4,
        rowHeights=[HEADER_ROW_HEIGHT],
        hAlign="LEFT",
    )
    dates_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), ACCENT),
                ("BOX", (0, 0), (-1, -1), 0.8, ACCENT_DARK),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    story.append(dates_tbl)
    story.append(Spacer(1, 10))

    ref_tbl = Table(
        [[p("", styles["SMALL"]), p("", styles["TEXT"])]],
        colWidths=[4.2 * cm, TOTAL_WIDTH - 4.2 * cm],
        rowHeights=[1.5 * cm],
        hAlign="LEFT",
    )
    ref_tbl.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    story.append(ref_tbl)
    story.append(Spacer(1, 12))
    story.append(p(labels["services_section"], styles["SMALL"]))
    story.append(Spacer(1, 6))

    rows = [
        [
            p(labels["service_type"], styles["WHITE_HEAD"]),
            p(labels["price"], styles["WHITE_HEAD"]),
            p(labels["problem"], styles["WHITE_HEAD"]),
        ]
    ]

    items = DATA.get("items") or [{}]
    for item in items:
        rows.append(
            [
                p(item.get("service", ""), styles["TABLE_CELL"]),
                p(format_price(item.get("price", ""), language), styles["TABLE_CELL"]),
                p(item.get("issue", ""), styles["TABLE_CELL"]),
            ]
        )

    col_widths = [
        9.6 * cm,
        2.2 * cm,
        TOTAL_WIDTH - (9.6 * cm + 2.2 * cm),
    ]

    items_tbl = Table(
        rows,
        colWidths=col_widths,
        rowHeights=[HEADER_ROW_HEIGHT] + [2.5 * cm] * (len(rows) - 1),
        hAlign="LEFT",
    )
    items_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("BOX", (0, 0), (-1, 0), 0.8, ACCENT_DARK),

                # The black service header now uses exactly the same spacing
                # and vertical alignment as the black date header.
                ("LEFTPADDING", (0, 0), (-1, 0), 10),
                ("RIGHTPADDING", (0, 0), (-1, 0), 10),
                ("TOPPADDING", (0, 0), (-1, 0), 7),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
                ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),

                # Body rows keep their original appearance. There is no extra
                # manually inserted empty row, so only DATA["items"] is rendered.
                ("LEFTPADDING", (0, 1), (-1, -1), 8),
                ("RIGHTPADDING", (0, 1), (-1, -1), 8),
                ("TOPPADDING", (0, 1), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 7),
                ("VALIGN", (0, 1), (-1, -1), "TOP"),
            ]
        )
    )

    story.append(items_tbl)
    story.append(Spacer(1, 12))

    problem = DATA.get("problem", {})

    story.append(
        box(
            labels["summary"],
            [problem.get("summary", "—")],
            styles,
            width=TOTAL_WIDTH,
            body_height=7.5 * cm,
        )
    )
    story.append(Spacer(1, 8))

    story.append(
        box(
            labels["description"],
            [problem.get("description", "—")],
            styles,
            width=TOTAL_WIDTH,
            body_height=5.8 * cm,
        )
    )
    story.append(Spacer(1, 8))

    story.append(p(labels["statement"], styles["SMALL"]))
    story.append(Spacer(1, 16))

    sign = Table(
        [[p(labels["signature_note"], styles["SMALL"])]],
        colWidths=[6.6 * cm],
        hAlign="RIGHT",
    )
    sign.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, -1), 0.8, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(sign)
    story.append(Spacer(1, 12))

    page_callback = partial(
        draw_watermark_fullpage,
        title=labels["complaint_title"],
    )
    doc.build(
        story,
        onFirstPage=page_callback,
        onLaterPages=page_callback,
    )


def main() -> None:
    ensure_paths()
    register_fonts()

    generate_pdf("sk", OUTPUT_PDF_SK)
    generate_pdf("en", OUTPUT_PDF_EN)

    print(f"Generated: {OUTPUT_PDF_SK}")
    print(f"Generated: {OUTPUT_PDF_EN}")


if __name__ == "__main__":
    main()