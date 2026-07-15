import os
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import partial
from html import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_PDF_SK = os.path.normpath(os.path.join(BASE_DIR, "..", "invoice_sk.pdf"))
OUTPUT_PDF_EN = os.path.normpath(os.path.join(BASE_DIR, "..", "invoice_en.pdf"))

# Rovnake cesty ako v complaint.py:
# D:\\Spectline\\lib\\documents\\generator -> D:\\Spectline\\public\\...
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
LIGHT_PANEL = colors.HexColor("#E8E8E8")

TOTAL_WIDTH = A4[0] - 4 * cm
HEADER_ROW_HEIGHT = 30
MONEY_QUANT = Decimal("0.01")

TODAY = date.today()


# -----------------------------------------------------------------------------
# UDAJE FAKTURY
# Upravte hodnoty v tomto slovniku. Jeden beh vytvori SK aj EN PDF.
# -----------------------------------------------------------------------------
DATA = {
    "provider": {
        "name": "Spectline",
        "address": "",
        "country": "Slovensko",
        "ico": "",
        "dic": "",
        "ic_dph": "",
        "email": "info.spectline@gmail.com",
        "phone": "",
        "bank": "",
        "iban": "",
        "swift": "",
        "vat_payer": True,
    },
    "customer": {
        "name": "",
        "address": "",
        "country": "",
        "email": "",
        "phone": "",
        "ico": "",
        "dic": "",
        "ic_dph": "",
    },
    "invoice": {
        "invoice_number": TODAY.strftime("%Y%m%d"),
        "variable_symbol": TODAY.strftime("%Y%m%d"),
        "delivery_date": TODAY.strftime("%d.%m.%Y"),
        "issue_date": TODAY.strftime("%d.%m.%Y"),
        "due_date": (TODAY + timedelta(days=14)).strftime("%d.%m.%Y"),
        # Kluc sa automaticky prelozi do SK/EN.
        "payment_method_key": "bank_transfer",
        "payment_method": "",
        "delivery_method_key": "electronic",
        "delivery_method": "",
        "currency": "EUR",
    },
    "items": [
        {
            "description": "",
            "quantity": 1,
            # Pouzite unit_key alebo vlastny text v unit.
            "unit_key": "piece",
            "unit": "",
            "unit_price_net": 0,
            "vat_rate": 23,
        }
    ],
    "note": "",
}


LANGUAGES = {
    "sk": {
        "title": "FAKTÚRA",
        "provider": "POSKYTOVATEĽ",
        "customer": "KLIENT",
        "company": "SPOLOČNOSŤ",
        "name": "MENO",
        "address": "ADRESA",
        "email": "E-MAIL",
        "phone": "TEL.",
        "company_id": "IČO",
        "tax_id": "DIČ",
        "vat_id": "IČ DPH",
        "invoice_number": "Číslo faktúry",
        "delivery_date": "DÁTUM DODANIA",
        "issue_date": "DÁTUM VYSTAVENIA",
        "due_date": "DÁTUM SPLATNOSTI",
        "payment_method": "Forma úhrady",
        "delivery_method": "Spôsob dodania",
        "bank": "Banka",
        "iban": "IBAN",
        "swift": "SWIFT",
        "variable_symbol": "Variabilný symbol",
        "invoice_for": "Fakturujeme Vám:",
        "item_description": "Názov a popis položky",
        "quantity": "Množstvo",
        "unit": "Jednotka",
        "unit_price_net": "Jedn. cena\nbez DPH",
        "vat": "DPH",
        "total_gross": "Celkom\ns DPH",
        "vat_rate": "Sadzba\nDPH",
        "tax_base": "Základ dane",
        "vat_amount": "DPH",
        "total_with_vat": "Celkom s DPH",
        "amount_due": "Suma na úhradu:",
        "vat_payer": "Dodávateľ je platiteľom DPH.",
        "not_vat_payer": "Dodávateľ nie je platiteľom DPH.",
        "signature": "Podpis a pečiatka",
        "note": "Poznámka",
        "item_placeholder": "Názov a popis položky",
        "payment_methods": {
            "bank_transfer": "Bankovým prevodom",
            "cash": "V hotovosti",
            "card": "Platobnou kartou",
        },
        "delivery_methods": {
            "electronic": "Elektronicky",
            "personal": "Osobne",
            "courier": "Kuriérom",
        },
        "units": {
            "piece": "ks",
            "hour": "hod.",
            "day": "deň",
            "project": "projekt",
            "service": "služba",
        },
    },
    "en": {
        "title": "INVOICE",
        "provider": "PROVIDER",
        "customer": "CLIENT",
        "company": "COMPANY",
        "name": "NAME",
        "address": "ADDRESS",
        "email": "E-MAIL",
        "phone": "TEL.",
        "company_id": "COMPANY ID",
        "tax_id": "TAX ID",
        "vat_id": "VAT ID",
        "invoice_number": "Invoice number",
        "delivery_date": "DELIVERY DATE",
        "issue_date": "ISSUE DATE",
        "due_date": "DUE DATE",
        "payment_method": "Payment method",
        "delivery_method": "Delivery method",
        "bank": "Bank",
        "iban": "IBAN",
        "swift": "SWIFT",
        "variable_symbol": "Variable symbol",
        "invoice_for": "We invoice you for:",
        "item_description": "Item name and description",
        "quantity": "Quantity",
        "unit": "Unit",
        "unit_price_net": "Unit price\nexcl. VAT",
        "vat": "VAT",
        "total_gross": "Total\nincl. VAT",
        "vat_rate": "VAT\nrate",
        "tax_base": "Tax base",
        "vat_amount": "VAT",
        "total_with_vat": "Total incl. VAT",
        "amount_due": "Amount due:",
        "vat_payer": "The supplier is registered for VAT.",
        "not_vat_payer": "The supplier is not registered for VAT.",
        "signature": "Signature and stamp",
        "note": "Note",
        "item_placeholder": "Item name and description",
        "payment_methods": {
            "bank_transfer": "Bank transfer",
            "cash": "Cash",
            "card": "Payment card",
        },
        "delivery_methods": {
            "electronic": "Electronically",
            "personal": "In person",
            "courier": "Courier",
        },
        "units": {
            "piece": "pc",
            "hour": "hr",
            "day": "day",
            "project": "project",
            "service": "service",
        },
    },
}


def ensure_paths() -> None:
    if not os.path.isfile(FONT_PATH):
        raise FileNotFoundError(f"Missing font: {FONT_PATH}")
    if not os.path.isfile(LOGO_PATH):
        raise FileNotFoundError(f"Missing logo: {LOGO_PATH}")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Roboto", FONT_PATH))


def as_decimal(value) -> Decimal:
    if value in (None, ""):
        return Decimal("0")

    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value).replace(" ", "").replace(",", "."))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid numeric value: {value!r}") from exc


def money(value) -> Decimal:
    return as_decimal(value).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)


def format_number(value, language: str, decimals: int = 2) -> str:
    number = as_decimal(value)
    quant = Decimal("1") if decimals == 0 else Decimal("1." + ("0" * decimals))
    number = number.quantize(quant, rounding=ROUND_HALF_UP)

    formatted = f"{number:,.{decimals}f}"
    if language == "sk":
        return formatted.replace(",", "X").replace(".", ",").replace("X", " ")
    return formatted


def format_quantity(value, language: str) -> str:
    number = as_decimal(value)
    if number == number.to_integral_value():
        return format_number(number, language, decimals=0)
    return format_number(number, language, decimals=2).rstrip("0").rstrip(",.")


def format_money(value, language: str, currency: str = "EUR", show_currency: bool = False) -> str:
    formatted = format_number(money(value), language, decimals=2)
    if show_currency:
        return f"{formatted} {currency}"
    return formatted


def format_percent(value, language: str) -> str:
    number = as_decimal(value)
    if number == number.to_integral_value():
        return f"{format_number(number, language, 0)}%"
    return f"{format_number(number, language, 2).rstrip('0').rstrip(',.')}%"


def display(value, fallback: str = "-") -> str:
    text = "" if value is None else str(value).strip()
    return text if text else fallback


def p(text: str, style: ParagraphStyle) -> Paragraph:
    safe = escape(str(text), quote=False).replace("\n", "<br/>")
    return Paragraph(safe, style)


def create_styles() -> dict:
    return {
        "TEXT": ParagraphStyle(
            "TEXT",
            fontName="Roboto",
            fontSize=9.8,
            leading=13,
            textColor=INK,
            spaceBefore=0,
            spaceAfter=0,
        ),
        "SMALL": ParagraphStyle(
            "SMALL",
            fontName="Roboto",
            fontSize=9,
            leading=12,
            textColor=MUTED,
            spaceBefore=0,
            spaceAfter=0,
        ),
        "BOX_HEAD": ParagraphStyle(
            "BOX_HEAD",
            fontName="Roboto",
            fontSize=9,
            leading=11,
            textColor=ACCENT_DARK,
            spaceBefore=0,
            spaceAfter=0,
        ),
        "WHITE_HEAD": ParagraphStyle(
            "WHITE_HEAD",
            fontName="Roboto",
            fontSize=8.5,
            leading=10.5,
            textColor=colors.white,
            spaceBefore=0,
            spaceAfter=0,
        ),
        "TABLE_CELL": ParagraphStyle(
            "TABLE_CELL",
            fontName="Roboto",
            fontSize=9,
            leading=11,
            textColor=INK,
            spaceBefore=0,
            spaceAfter=0,
        ),
        "AMOUNT_LABEL": ParagraphStyle(
            "AMOUNT_LABEL",
            fontName="Roboto",
            fontSize=10,
            leading=12,
            textColor=colors.white,
            spaceBefore=0,
            spaceAfter=0,
        ),
        "AMOUNT_VALUE": ParagraphStyle(
            "AMOUNT_VALUE",
            fontName="Roboto",
            fontSize=17,
            leading=19,
            textColor=colors.white,
            alignment=2,
            spaceBefore=0,
            spaceAfter=0,
        ),
    }


def draw_watermark_fullpage(canv, doc, title: str) -> None:
    width, height = A4

    canv.saveState()
    try:
        if hasattr(canv, "setFillAlpha"):
            canv.setFillAlpha(0.06)
    except Exception:
        pass

    image = ImageReader(LOGO_PATH)
    image_width, image_height = image.getSize()
    scale = min(width / image_width, height / image_height) * 0.88
    draw_width = image_width * scale
    draw_height = image_height * scale
    x = (width - draw_width) / 2.0
    y = (height - draw_height) / 2.0

    canv.drawImage(
        image,
        x,
        y,
        width=draw_width,
        height=draw_height,
        mask="auto",
        preserveAspectRatio=True,
    )
    canv.restoreState()

    canv.saveState()
    canv.setFont("Roboto", 28)
    canv.setFillColor(ACCENT)
    canv.drawRightString(width - 2 * cm, height - 2.2 * cm, title)
    canv.restoreState()


def info_box(title: str, lines: list[str], styles: dict, width: float) -> Table:
    rows = [[p(title, styles["BOX_HEAD"])]]
    rows.extend([[p(line, styles["TEXT"])]] for line in lines)

    table = Table(
        rows,
        colWidths=[width],
        rowHeights=[16] + [None] * len(lines),
        hAlign="LEFT",
        splitByRow=0,
    )
    table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, LINE),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, 0), 4),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
                ("TOPPADDING", (0, 1), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def translated_choice(data: dict, value_key: str, text_key: str, translations: dict) -> str:
    custom_text = str(data.get(text_key, "") or "").strip()
    if custom_text:
        return custom_text

    choice_key = str(data.get(value_key, "") or "").strip()
    return translations.get(choice_key, choice_key)


def item_unit(item: dict, labels: dict) -> str:
    custom_unit = str(item.get("unit", "") or "").strip()
    if custom_unit:
        return custom_unit

    unit_key = str(item.get("unit_key", "piece") or "piece")
    return labels["units"].get(unit_key, unit_key)


def calculate_items(items: list[dict]) -> tuple[list[dict], dict[Decimal, dict], Decimal]:
    calculated_items = []
    vat_summary: dict[Decimal, dict] = {}
    grand_total = Decimal("0")

    for raw_item in items or [{}]:
        quantity = as_decimal(raw_item.get("quantity", 0))
        unit_price_net = money(raw_item.get("unit_price_net", 0))
        vat_rate = as_decimal(raw_item.get("vat_rate", 0))

        net_total = money(quantity * unit_price_net)
        vat_total = money(net_total * vat_rate / Decimal("100"))
        gross_total = money(net_total + vat_total)

        calculated = dict(raw_item)
        calculated.update(
            {
                "quantity_decimal": quantity,
                "unit_price_net_decimal": unit_price_net,
                "vat_rate_decimal": vat_rate,
                "net_total": net_total,
                "vat_total": vat_total,
                "gross_total": gross_total,
            }
        )
        calculated_items.append(calculated)

        if vat_rate not in vat_summary:
            vat_summary[vat_rate] = {
                "net_total": Decimal("0"),
                "vat_total": Decimal("0"),
                "gross_total": Decimal("0"),
            }

        vat_summary[vat_rate]["net_total"] += net_total
        vat_summary[vat_rate]["vat_total"] += vat_total
        vat_summary[vat_rate]["gross_total"] += gross_total
        grand_total += gross_total

    for summary in vat_summary.values():
        summary["net_total"] = money(summary["net_total"])
        summary["vat_total"] = money(summary["vat_total"])
        summary["gross_total"] = money(summary["gross_total"])

    return calculated_items, vat_summary, money(grand_total)


def build_party_section(labels: dict, styles: dict) -> Table:
    provider = DATA["provider"]
    customer = DATA["customer"]
    half_width = TOTAL_WIDTH / 2.0

    provider_box = info_box(
        labels["provider"],
        [
            f"{labels['company']}: {display(provider.get('name'))}",
            f"{labels['address']}: {display(provider.get('address'))}",
            f"{labels['company_id']}: {display(provider.get('ico'))}",
            f"{labels['tax_id']}: {display(provider.get('dic'))}",
            f"{labels['vat_id']}: {display(provider.get('ic_dph'))}",
            f"{labels['email']}: {display(provider.get('email'))}",
            f"{labels['phone']}: {display(provider.get('phone'))}",
        ],
        styles,
        half_width,
    )

    customer_box = info_box(
        labels["customer"],
        [
            f"{labels['name']}: {display(customer.get('name'))}",
            f"{labels['address']}: {display(customer.get('address'))}",
            f"{labels['email']}: {display(customer.get('email'))}",
            f"{labels['phone']}: {display(customer.get('phone'))}",
            f"{labels['company_id']}: {display(customer.get('ico'))}",
            f"{labels['tax_id']}: {display(customer.get('dic'))}",
            f"{labels['vat_id']}: {display(customer.get('ic_dph'))}",
        ],
        styles,
        half_width,
    )

    outer = Table(
        [[provider_box, customer_box]],
        colWidths=[half_width, half_width],
        hAlign="LEFT",
    )
    outer.setStyle(
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
    return outer


def build_dates_bar(labels: dict, styles: dict) -> Table:
    invoice = DATA["invoice"]
    rows = [
        [
            p(
                f"{labels['delivery_date']}: {display(invoice.get('delivery_date'))}",
                styles["WHITE_HEAD"],
            ),
            p(
                f"{labels['issue_date']}: {display(invoice.get('issue_date'))}",
                styles["WHITE_HEAD"],
            ),
            p(
                f"{labels['due_date']}: {display(invoice.get('due_date'))}",
                styles["WHITE_HEAD"],
            ),
        ]
    ]

    table = Table(
        rows,
        colWidths=[TOTAL_WIDTH / 3.0] * 3,
        rowHeights=[HEADER_ROW_HEIGHT],
        hAlign="LEFT",
    )
    table.setStyle(
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
    return table


def build_payment_details(labels: dict, styles: dict) -> Table:
    provider = DATA["provider"]
    invoice = DATA["invoice"]

    payment_method = translated_choice(
        invoice,
        "payment_method_key",
        "payment_method",
        labels["payment_methods"],
    )
    delivery_method = translated_choice(
        invoice,
        "delivery_method_key",
        "delivery_method",
        labels["delivery_methods"],
    )

    details = [
        (labels["invoice_number"], invoice.get("invoice_number")),
        (labels["payment_method"], payment_method),
        (labels["delivery_method"], delivery_method),
        (labels["bank"], provider.get("bank")),
        (labels["iban"], provider.get("iban")),
        (labels["swift"], provider.get("swift")),
        (labels["variable_symbol"], invoice.get("variable_symbol")),
    ]

    rows = [
        [p(label, styles["SMALL"]), p(display(value), styles["TEXT"])]
        for label, value in details
    ]

    table = Table(
        rows,
        colWidths=[4.2 * cm, TOTAL_WIDTH - 4.2 * cm],
        hAlign="LEFT",
    )
    table.setStyle(
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
    return table


def build_items_table(
    labels: dict,
    styles: dict,
    language: str,
    calculated_items: list[dict],
) -> Table:
    rows = [
        [
            p(labels["item_description"], styles["WHITE_HEAD"]),
            p(labels["quantity"], styles["WHITE_HEAD"]),
            p(labels["unit"], styles["WHITE_HEAD"]),
            p(labels["unit_price_net"], styles["WHITE_HEAD"]),
            p(labels["vat"], styles["WHITE_HEAD"]),
            p(labels["total_gross"], styles["WHITE_HEAD"]),
        ]
    ]

    for item in calculated_items:
        rows.append(
            [
                p(
                    display(item.get("description"), labels["item_placeholder"]),
                    styles["TABLE_CELL"],
                ),
                p(format_quantity(item["quantity_decimal"], language), styles["TABLE_CELL"]),
                p(item_unit(item, labels), styles["TABLE_CELL"]),
                p(format_money(item["unit_price_net_decimal"], language), styles["TABLE_CELL"]),
                p(format_percent(item["vat_rate_decimal"], language), styles["TABLE_CELL"]),
                p(format_money(item["gross_total"], language), styles["TABLE_CELL"]),
            ]
        )

    col_widths = [
        6.8 * cm,
        2.0 * cm,
        1.8 * cm,
        2.8 * cm,
        1.4 * cm,
        TOTAL_WIDTH - (6.8 * cm + 2.0 * cm + 1.8 * cm + 2.8 * cm + 1.4 * cm),
    ]

    table = Table(
        rows,
        colWidths=col_widths,
        rowHeights=[HEADER_ROW_HEIGHT] + [1.2 * cm] * (len(rows) - 1),
        hAlign="LEFT",
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("BOX", (0, 0), (-1, -1), 0.8, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("LEFTPADDING", (0, 0), (-1, 0), 8),
                ("RIGHTPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, 0), 6),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
                ("LEFTPADDING", (0, 1), (-1, -1), 8),
                ("RIGHTPADDING", (0, 1), (-1, -1), 8),
                ("TOPPADDING", (0, 1), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 7),
                ("VALIGN", (0, 1), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def build_totals_section(
    labels: dict,
    styles: dict,
    language: str,
    vat_summary: dict[Decimal, dict],
    grand_total: Decimal,
) -> Table:
    currency = str(DATA["invoice"].get("currency", "EUR") or "EUR")

    summary_rows = [
        [
            p(labels["vat_rate"], styles["SMALL"]),
            p(labels["tax_base"], styles["SMALL"]),
            p(labels["vat_amount"], styles["SMALL"]),
            p(labels["total_with_vat"], styles["SMALL"]),
        ]
    ]

    for vat_rate in sorted(vat_summary.keys()):
        totals = vat_summary[vat_rate]
        summary_rows.append(
            [
                p(format_percent(vat_rate, language), styles["TABLE_CELL"]),
                p(format_money(totals["net_total"], language, currency, True), styles["TABLE_CELL"]),
                p(format_money(totals["vat_total"], language, currency, True), styles["TABLE_CELL"]),
                p(format_money(totals["gross_total"], language, currency, True), styles["TABLE_CELL"]),
            ]
        )

    summary_width = 9.8 * cm
    amount_width = TOTAL_WIDTH - summary_width - 0.4 * cm

    summary_table = Table(
        summary_rows,
        colWidths=[2.1 * cm, 2.8 * cm, 2.1 * cm, summary_width - 7.0 * cm],
        hAlign="LEFT",
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_PANEL),
                ("BOX", (0, 0), (-1, -1), 0.8, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    amount_table = Table(
        [
            [
                p(labels["amount_due"], styles["AMOUNT_LABEL"]),
                p(format_money(grand_total, language, currency, True), styles["AMOUNT_VALUE"]),
            ]
        ],
        colWidths=[amount_width * 0.40, amount_width * 0.60],
        rowHeights=[2.0 * cm],
        hAlign="LEFT",
    )
    amount_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), ACCENT),
                ("BOX", (0, 0), (-1, -1), 0.8, ACCENT_DARK),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    outer = Table(
        [[summary_table, "", amount_table]],
        colWidths=[summary_width, 0.4 * cm, amount_width],
        hAlign="LEFT",
    )
    outer.setStyle(
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
    return outer


def build_signature(labels: dict, styles: dict) -> Table:
    signature = Table(
        [[p(labels["signature"], styles["SMALL"]) ]],
        colWidths=[6.2 * cm],
        hAlign="RIGHT",
    )
    signature.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, -1), 0.8, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return signature


def generate_pdf(language: str, output_pdf: str) -> None:
    labels = LANGUAGES[language]
    styles = create_styles()
    calculated_items, vat_summary, grand_total = calculate_items(DATA.get("items") or [{}])

    output_dir = os.path.dirname(output_pdf)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.7 * cm,
        bottomMargin=1.8 * cm,
        title=labels["title"],
        author=DATA["provider"].get("name", "Spectline"),
    )

    story = [Spacer(1, 8)]
    story.append(build_party_section(labels, styles))
    story.append(Spacer(1, 10))
    story.append(build_dates_bar(labels, styles))
    story.append(Spacer(1, 10))
    story.append(build_payment_details(labels, styles))
    story.append(Spacer(1, 10))
    story.append(p(labels["invoice_for"], styles["SMALL"]))
    story.append(Spacer(1, 6))
    story.append(build_items_table(labels, styles, language, calculated_items))
    story.append(Spacer(1, 10))
    story.append(
        KeepTogether(
            [
                build_totals_section(labels, styles, language, vat_summary, grand_total),
                Spacer(1, 10),
            ]
        )
    )

    vat_message = labels["vat_payer"] if DATA["provider"].get("vat_payer") else labels["not_vat_payer"]
    story.append(p(vat_message, styles["SMALL"]))

    note = str(DATA.get("note", "") or "").strip()
    if note:
        story.append(Spacer(1, 7))
        story.append(p(f"{labels['note']}: {note}", styles["SMALL"]))

    story.append(Spacer(1, 20))
    story.append(build_signature(labels, styles))

    page_callback = partial(draw_watermark_fullpage, title=labels["title"])
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