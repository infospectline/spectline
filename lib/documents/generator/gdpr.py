# public/documents/generator/gdpr.py
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

OUTPUT_PDF_SK = os.path.normpath(os.path.join(BASE_DIR, "..", "gdpr_sk.pdf"))
OUTPUT_PDF_EN = os.path.normpath(os.path.join(BASE_DIR, "..", "gdpr_en.pdf"))

# Expected project structure:
# public/
#   documents/generator/gdpr.py
#   images/logo_pdf.png
#   fonts/Roboto-Regular.ttf
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

CONTENT_WIDTH = A4[0] - 4 * cm


# IMPORTANT:
# Before publishing the generated PDFs, complete the full legal identity of the
# controller below. A brand name and email address alone may not be sufficient
# for a complete Article 13 GDPR notice.
DATA = {
    "version": "2.0",
    "website": "spectline.com",
    "controller": {
        "brand_name": "Spectline",
        "legal_name": "Spectline",
        "address": "",
        "country_sk": "Slovenská republika",
        "country_en": "Slovak Republic",
        "ico": "",
        "dic": "",
        "ic_dph": "",
        "email": "info.spectline@gmail.com",
    },
    # These are operational defaults. Adjust them to match the real systems
    # and internal retention policy used by Spectline.
    "retention": {
        "inquiry_months": 12,
        "technical_logs_days": 90,
    },
}


LANGUAGES = {
    "sk": {
        "title": "OCHRANA OSOBNÝCH ÚDAJOV (GDPR)",
        "intro": (
            "Spectline je kreatívne štúdio pre VFX, animáciu, 3D, motion dizajn "
            "a postprodukciu. V tomto dokumente vysvetľujeme, aké osobné údaje "
            "spracúvame pri používaní webu, komunikácii o projektoch, poskytovaní "
            "služieb a odbere noviniek, na aké účely ich používame a aké práva "
            "majú dotknuté osoby."
        ),
        "meta_updated": "Posledná aktualizácia",
        "meta_version": "Verzia",
        "meta_site": "Web",
        "footer_page": "Strana",
        "controller_labels": {
            "legal_name": "Prevádzkovateľ",
            "brand_name": "Obchodné označenie",
            "address": "Sídlo / adresa",
            "country": "Krajina",
            "ico": "IČO",
            "dic": "DIČ",
            "ic_dph": "IČ DPH",
            "email": "Kontaktný e-mail",
            "website": "Web",
        },
    },
    "en": {
        "title": "PERSONAL DATA PROTECTION (GDPR)",
        "intro": (
            "Spectline is a creative studio providing VFX, animation, 3D, motion "
            "design and post-production services. This notice explains what personal "
            "data we process when you use our website, contact us about a project, "
            "use our services or subscribe to studio updates, why we use the data "
            "and what rights data subjects have."
        ),
        "meta_updated": "Last updated",
        "meta_version": "Version",
        "meta_site": "Website",
        "footer_page": "Page",
        "controller_labels": {
            "legal_name": "Controller",
            "brand_name": "Trading name",
            "address": "Registered office / address",
            "country": "Country",
            "ico": "Company ID",
            "dic": "Tax ID",
            "ic_dph": "VAT ID",
            "email": "Contact email",
            "website": "Website",
        },
    },
}


def current_date(language: str) -> str:
    today = date.today()
    if language == "sk":
        return today.strftime("%d.%m.%Y")
    return today.strftime("%d %B %Y")


def controller_value(key: str, language: str) -> str:
    controller = DATA["controller"]
    if key == "country":
        return controller["country_sk"] if language == "sk" else controller["country_en"]
    if key == "website":
        return DATA["website"]
    return str(controller.get(key, "")).strip()


def controller_lines(language: str) -> list[str]:
    labels = LANGUAGES[language]["controller_labels"]
    controller = DATA["controller"]

    ordered_keys = [
        "legal_name",
        "brand_name",
        "address",
        "country",
        "ico",
        "dic",
        "ic_dph",
        "email",
        "website",
    ]

    lines: list[str] = []
    for key in ordered_keys:
        value = controller_value(key, language)
        if not value:
            continue

        # Avoid repeating the same name twice when legal and trading names match.
        if (
            key == "brand_name"
            and value.casefold() == str(controller.get("legal_name", "")).strip().casefold()
        ):
            continue

        lines.append(f"{labels[key]}: {value}")

    return lines


def build_sections(language: str) -> list[dict]:
    email = DATA["controller"]["email"]
    website = DATA["website"]
    inquiry_months = DATA["retention"]["inquiry_months"]
    technical_logs_days = DATA["retention"]["technical_logs_days"]

    if language == "sk":
        return [
            {
                "title": "1. Prevádzkovateľ a kontakt",
                "paragraphs": [
                    "Prevádzkovateľom osobných údajov je subjekt uvedený nižšie. Prevádzkovateľ určuje účely a prostriedky spracúvania osobných údajov.",
                ],
                "bullets": controller_lines("sk"),
            },
            {
                "title": "2. Aké osobné údaje spracúvame",
                "paragraphs": [
                    "Spracúvame iba údaje primerané konkrétnemu účelu. Rozsah údajov závisí od toho, ako s nami komunikujete a aké služby využívate.",
                ],
                "bullets": [
                    "Kontaktné a identifikačné údaje, najmä meno, priezvisko, názov spoločnosti, e-mail, telefón a fakturačné údaje.",
                    "Údaje z projektového dopytu, najmä typ projektu, rozsah, termín, rozpočet, požiadavky a obsah komunikácie.",
                    "Zmluvné, objednávkové, platobné a účtovné údaje potrebné na cenovú ponuku, realizáciu projektu a fakturáciu.",
                    "Projektové súbory a podklady, napríklad video, zvuk, fotografie, grafika, 3D dáta, mená, podobizne alebo hlas, ak ich klient poskytne na spracovanie.",
                    "Údaje súvisiace s portfóliom alebo showreelom, ak bolo ich použitie dohodnuté alebo existuje iný platný právny základ.",
                    "Technické údaje, napríklad IP adresa, typ zariadenia, prehliadač, systémové logy a informácie o cookies.",
                    "E-mailová adresa a záznam o súhlase pri prihlásení na odber noviniek.",
                ],
            },
            {
                "title": "3. Účely a právne základy spracúvania",
                "paragraphs": [
                    "Osobné údaje spracúvame len na konkrétne účely a na základe niektorého z právnych základov podľa článku 6 GDPR.",
                ],
                "bullets": [
                    "Odpoveď na dopyt, príprava cenovej ponuky a kroky pred uzatvorením zmluvy - plnenie opatrení pred uzatvorením zmluvy alebo oprávnený záujem na obchodnej komunikácii.",
                    "Realizácia VFX, animačných, 3D, postprodukčných a súvisiacich kreatívnych služieb - plnenie zmluvy.",
                    "Fakturácia, vedenie účtovníctva a plnenie daňových alebo iných zákonných povinností - plnenie zákonnej povinnosti.",
                    "Ochrana webu, systémov, projektových súborov a prevencia podvodov alebo zneužitia - oprávnený záujem na bezpečnej prevádzke.",
                    "Ochrana, uplatnenie alebo obhajoba právnych nárokov - oprávnený záujem.",
                    "Zasielanie newslettera a nepovinné analytické alebo marketingové cookies - súhlas, ktorý možno kedykoľvek odvolať.",
                    "Zverejnenie práce v portfóliu alebo showreeli - zmluva, licencia, súhlas alebo oprávnený záujem podľa konkrétnej dohody a povahy materiálu.",
                ],
            },
            {
                "title": "4. Kontaktný formulár a projektové dopyty",
                "paragraphs": [
                    "Kontaktný formulár na webe môže požadovať meno, typ projektu, e-mail a detaily projektu. Poskytnutie údajov je dobrovoľné, ale bez kontaktných údajov a základného opisu projektu nemusíme vedieť na dopyt odpovedať alebo pripraviť ponuku.",
                    "Ak dopyt nevedie k spolupráci, komunikáciu spravidla uchovávame najviac {months} mesiacov od posledného kontaktu, pokiaľ dlhšie uchovanie nie je potrebné na ochranu právnych nárokov alebo z iného zákonného dôvodu.".format(months=inquiry_months),
                ],
            },
            {
                "title": "5. Projektové materiály a zodpovednosť klienta",
                "paragraphs": [
                    "Pri realizácii projektu môžeme spracúvať materiály dodané klientom. Klient je zodpovedný za to, že má oprávnenie tieto materiály poskytnúť a že boli splnené potrebné informačné povinnosti, súhlasy, licencie alebo iné právne podmienky voči osobám zachyteným v materiáloch.",
                    "Projektové materiály používame len na dohodnutú produkciu, komunikáciu, odovzdanie výstupov, technickú archiváciu a prípadné portfólio, ak bolo také použitie osobitne dohodnuté alebo je inak zákonné.",
                ],
            },
            {
                "title": "6. Newsletter a priama komunikácia",
                "paragraphs": [
                    "Na odber noviniek zo štúdia používame e-mailovú adresu na základe súhlasu. Súhlas môžete kedykoľvek odvolať prostredníctvom odkazu na odhlásenie v správe alebo e-mailom na {email}. Odvolanie súhlasu nemá vplyv na zákonnosť spracúvania vykonaného pred jeho odvolaním.".format(email=email),
                    "Po odhlásení môžeme v nevyhnutnom rozsahu uchovať záznam o udelení a odvolaní súhlasu, aby sme vedeli preukázať splnenie právnych povinností a zabránili ďalšiemu zasielaniu.",
                ],
            },
            {
                "title": "7. Cookies a technické údaje",
                "paragraphs": [
                    "Web môže používať nevyhnutné technické cookies potrebné na bezpečnosť a základné fungovanie stránky. Takéto cookies sa používajú bez súhlasu, ak sú skutočne nevyhnutné.",
                    "Analytické, preferenčné alebo marketingové cookies sa použijú iba po udelení súhlasu, ak sú na webe nasadené. Súhlas možno zmeniť alebo odvolať v nastaveniach cookies. Odstránenie cookies je možné aj v nastaveniach prehliadača.",
                    "Technické logy sa pri bežnej prevádzke uchovávajú spravidla najviac {days} dní, pokiaľ nie sú potrebné dlhšie na vyšetrovanie bezpečnostného incidentu, ochranu práv alebo splnenie zákonnej povinnosti.".format(days=technical_logs_days),
                ],
            },
            {
                "title": "8. Príjemcovia a sprostredkovatelia",
                "paragraphs": [
                    "Osobné údaje neposkytujeme tretím stranám na ich vlastné marketingové účely. V nevyhnutnom rozsahu ich však môžu spracúvať dôveryhodní dodávatelia alebo partneri, ktorí nám pomáhajú prevádzkovať štúdio a poskytovať služby.",
                ],
                "bullets": [
                    "Poskytovatelia webhostingu, domény, e-mailu a technickej infraštruktúry.",
                    "Poskytovatelia cloudového úložiska, prenosu súborov, projektovej komunikácie a zálohovania.",
                    "Účtovníci, daňoví poradcovia, banky, platobné a fakturačné služby.",
                    "Dodávatelia newslettera, analytiky alebo správy cookies, ak sa tieto služby používajú.",
                    "Externí spolupracovníci zapojení do projektu, ak je to potrebné a zmluvne zabezpečené.",
                    "Právni poradcovia, poisťovne, súdy, úrady alebo orgány verejnej moci, ak to vyžaduje zákon alebo ochrana práv.",
                ],
            },
            {
                "title": "9. Prenos údajov mimo EÚ/EHP",
                "paragraphs": [
                    "Niektorí technickí dodávatelia môžu spracúvať údaje mimo Európskej únie alebo Európskeho hospodárskeho priestoru. Taký prenos uskutočníme len vtedy, ak je založený na rozhodnutí Európskej komisie o primeranosti alebo na vhodných zárukách, najmä štandardných zmluvných doložkách, prípadne ďalších ochranných opatreniach vyžadovaných GDPR.",
                ],
            },
            {
                "title": "10. Doba uchovávania",
                "paragraphs": [
                    "Osobné údaje uchovávame len počas obdobia potrebného na príslušný účel. Projektové a zmluvné údaje uchovávame počas spolupráce a následne počas platných premlčacích lehôt. Účtovné a daňové doklady uchovávame počas obdobia vyžadovaného príslušnými právnymi predpismi.",
                    "Po uplynutí doby uchovávania údaje bezpečne vymažeme, anonymizujeme alebo ich ďalej uchovávame len vtedy, ak to vyžaduje zákon, riešenie sporu, bezpečnostný incident alebo ochrana právnych nárokov.",
                ],
            },
            {
                "title": "11. Práva dotknutých osôb",
                "paragraphs": [
                    "Za podmienok stanovených GDPR máte právo požiadať o prístup k osobným údajom, ich opravu, výmaz, obmedzenie spracúvania a prenosnosť údajov. Máte tiež právo namietať proti spracúvaniu založenému na oprávnenom záujme a kedykoľvek odvolať súhlas.",
                    "Žiadosť môžete poslať na {email}. Pred vybavením žiadosti môžeme primeraným spôsobom overiť vašu totožnosť. Na žiadosť odpovieme v lehote stanovenej GDPR.".format(email=email),
                    "Ak sa domnievate, že spracúvanie porušuje právne predpisy, máte právo podať návrh alebo sťažnosť na Úrad na ochranu osobných údajov Slovenskej republiky, dataprotection.gov.sk, alebo na iný príslušný dozorný orgán.",
                ],
            },
            {
                "title": "12. Automatizované rozhodovanie",
                "paragraphs": [
                    "Spectline nevykonáva rozhodovanie založené výlučne na automatizovanom spracúvaní, ktoré by malo voči vám právne účinky alebo vás podobne významne ovplyvňovalo. Nevykonávame ani takéto profilovanie.",
                ],
            },
            {
                "title": "13. Bezpečnosť osobných údajov",
                "paragraphs": [
                    "Používame primerané technické a organizačné opatrenia na ochranu údajov pred neoprávneným prístupom, stratou, zmenou alebo zverejnením. Opatrenia môžu zahŕňať riadenie prístupov, zabezpečený prenos, zálohovanie, aktualizácie systémov, obmedzenie rozsahu údajov a zmluvné záväzky osôb, ktoré s údajmi pracujú.",
                    "Žiadny spôsob prenosu alebo uloženia údajov nie je úplne bez rizika. Bezpečnostné opatrenia preto pravidelne primerane prehodnocujeme podľa povahy spracúvania.",
                ],
            },
            {
                "title": "14. Zmeny dokumentu",
                "paragraphs": [
                    "Tento dokument môžeme aktualizovať najmä pri zmene služieb, dodávateľov, používaných technológií alebo právnych požiadaviek. Aktuálna verzia bude dostupná na webe {website}.".format(website=website),
                    "Posledná aktualizácia: {updated}.".format(updated=current_date("sk")),
                ],
            },
        ]

    return [
        {
            "title": "1. Controller and contact details",
            "paragraphs": [
                "The entity identified below is the controller of personal data and determines the purposes and means of processing.",
            ],
            "bullets": controller_lines("en"),
        },
        {
            "title": "2. Personal data we process",
            "paragraphs": [
                "We process only data that is appropriate for a specific purpose. The scope depends on how you communicate with us and which services you use.",
            ],
            "bullets": [
                "Contact and identification data, such as name, company name, email address, telephone number and billing details.",
                "Project inquiry data, such as project type, scope, deadline, budget, requirements and the content of communications.",
                "Contract, order, payment and accounting data required to prepare a quotation, deliver a project and issue invoices.",
                "Project files and source materials, including video, audio, photographs, graphics, 3D data, names, likenesses or voices where supplied by the client for processing.",
                "Information connected with a portfolio or showreel where its use has been agreed or another valid legal basis applies.",
                "Technical information, such as IP address, device type, browser, system logs and cookie information.",
                "Email address and consent records when subscribing to studio updates.",
            ],
        },
        {
            "title": "3. Purposes and legal bases",
            "paragraphs": [
                "We process personal data only for specified purposes and on one or more legal bases under Article 6 GDPR.",
            ],
            "bullets": [
                "Responding to inquiries, preparing quotations and taking steps before entering into a contract - pre-contractual measures or our legitimate interest in business communication.",
                "Providing VFX, animation, 3D, post-production and related creative services - performance of a contract.",
                "Invoicing, accounting and compliance with tax or other statutory obligations - compliance with a legal obligation.",
                "Protecting the website, systems and project files and preventing fraud or misuse - our legitimate interest in secure operations.",
                "Establishing, exercising or defending legal claims - our legitimate interest.",
                "Sending newsletters and using optional analytics or marketing cookies - consent, which may be withdrawn at any time.",
                "Publishing work in a portfolio or showreel - contract, licence, consent or legitimate interest, depending on the specific agreement and material.",
            ],
        },
        {
            "title": "4. Contact form and project inquiries",
            "paragraphs": [
                "Our website contact form may request your name, project type, email address and project details. Providing the data is voluntary, but without contact details and a basic project description we may be unable to respond or prepare a quotation.",
                "Where an inquiry does not lead to a collaboration, we normally retain the communication for no longer than {months} months after the last contact, unless longer retention is necessary to protect legal claims or for another lawful reason.".format(months=inquiry_months),
            ],
        },
        {
            "title": "5. Project materials and client responsibility",
            "paragraphs": [
                "When delivering a project, we may process materials supplied by the client. The client is responsible for ensuring that it is authorised to provide those materials and that any required privacy information, consents, licences or other legal conditions have been addressed for individuals appearing in the materials.",
                "We use project materials only for the agreed production, communication, delivery, technical archiving and possible portfolio use where that use has been separately agreed or is otherwise lawful.",
            ],
        },
        {
            "title": "6. Newsletter and direct communication",
            "paragraphs": [
                "We use your email address to send studio updates on the basis of consent. You may withdraw consent at any time by using the unsubscribe link in a message or by emailing {email}. Withdrawal does not affect the lawfulness of processing carried out before withdrawal.".format(email=email),
                "After an unsubscribe request, we may keep a limited record of the grant and withdrawal of consent where necessary to demonstrate compliance and prevent further messages.",
            ],
        },
        {
            "title": "7. Cookies and technical data",
            "paragraphs": [
                "The website may use strictly necessary technical cookies required for security and basic website operation. Such cookies may be used without consent where they are genuinely necessary.",
                "Analytics, preference or marketing cookies are used only after consent where those technologies are deployed. Consent can be changed or withdrawn in the cookie settings. Cookies may also be removed through browser settings.",
                "During normal operation, technical logs are generally retained for no longer than {days} days unless they are required for longer to investigate a security incident, protect rights or comply with a legal obligation.".format(days=technical_logs_days),
            ],
        },
        {
            "title": "8. Recipients and processors",
            "paragraphs": [
                "We do not disclose personal data to third parties for their own marketing purposes. Trusted suppliers or partners may, however, process data to the extent necessary to operate the studio and deliver services.",
            ],
            "bullets": [
                "Website hosting, domain, email and technical infrastructure providers.",
                "Cloud storage, file transfer, project communication and backup providers.",
                "Accountants, tax advisers, banks, payment and invoicing providers.",
                "Newsletter, analytics or cookie management providers where those services are used.",
                "External collaborators involved in a project where necessary and contractually protected.",
                "Legal advisers, insurers, courts, regulators or public authorities where required by law or necessary to protect rights.",
            ],
        },
        {
            "title": "9. Transfers outside the EU/EEA",
            "paragraphs": [
                "Some technology providers may process data outside the European Union or European Economic Area. We make such transfers only where they are covered by a European Commission adequacy decision or appropriate safeguards, particularly Standard Contractual Clauses and any additional measures required by the GDPR.",
            ],
        },
        {
            "title": "10. Retention periods",
            "paragraphs": [
                "We retain personal data only for as long as necessary for the relevant purpose. Project and contract data are kept during the collaboration and afterwards for applicable limitation periods. Accounting and tax records are retained for the periods required by applicable law.",
                "When a retention period expires, data are securely deleted or anonymised unless continued retention is required by law, a dispute, a security incident or the protection of legal claims.",
            ],
        },
        {
            "title": "11. Data subject rights",
            "paragraphs": [
                "Subject to the conditions in the GDPR, you may request access to, rectification or erasure of your personal data, restriction of processing and data portability. You may also object to processing based on legitimate interests and withdraw consent at any time.",
                "Requests may be sent to {email}. Before responding, we may reasonably verify your identity. We will respond within the period required by the GDPR.".format(email=email),
                "If you believe that processing infringes data protection law, you may lodge a complaint or initiate proceedings with the Office for Personal Data Protection of the Slovak Republic at dataprotection.gov.sk, or with another competent supervisory authority.",
            ],
        },
        {
            "title": "12. Automated decision-making",
            "paragraphs": [
                "Spectline does not make decisions based solely on automated processing that produce legal effects concerning you or similarly significantly affect you. We do not carry out profiling of this kind.",
            ],
        },
        {
            "title": "13. Data security",
            "paragraphs": [
                "We use appropriate technical and organisational measures to protect data against unauthorised access, loss, alteration or disclosure. Measures may include access controls, secure transmission, backups, system updates, data minimisation and contractual obligations for persons handling the data.",
                "No method of transmission or storage is entirely risk-free. We therefore review security measures as appropriate to the nature of the processing.",
            ],
        },
        {
            "title": "14. Changes to this notice",
            "paragraphs": [
                "We may update this notice, particularly when our services, suppliers, technologies or legal requirements change. The current version will be available at {website}.".format(website=website),
                "Last updated: {updated}.".format(updated=current_date("en")),
            ],
        },
    ]


def ensure_paths() -> None:
    if not os.path.isfile(FONT_PATH):
        raise FileNotFoundError(f"Missing font: {FONT_PATH}")
    if not os.path.isfile(LOGO_PATH):
        raise FileNotFoundError(f"Missing logo: {LOGO_PATH}")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Roboto", FONT_PATH))


def escape_text(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def p(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape_text(text), style)


def create_styles() -> dict[str, ParagraphStyle]:
    return {
        "TEXT": ParagraphStyle(
            "TEXT",
            fontName="Roboto",
            fontSize=9.6,
            leading=13,
            textColor=INK,
            spaceBefore=0,
            spaceAfter=0,
        ),
        "SMALL": ParagraphStyle(
            "SMALL",
            fontName="Roboto",
            fontSize=8.6,
            leading=11,
            textColor=MUTED,
            spaceBefore=0,
            spaceAfter=0,
        ),
        "BOX_HEAD": ParagraphStyle(
            "BOX_HEAD",
            fontName="Roboto",
            fontSize=9.2,
            leading=12,
            textColor=ACCENT_DARK,
            spaceBefore=0,
            spaceAfter=0,
        ),
    }


def box(title: str, paragraphs: list[str], bullets: list[str], styles: dict) -> Table:
    rows = [[p(title, styles["BOX_HEAD"])]]

    for paragraph in paragraphs:
        if str(paragraph).strip():
            rows.append([p(paragraph, styles["TEXT"])])

    for bullet in bullets:
        if str(bullet).strip():
            rows.append([p(f"• {bullet}", styles["TEXT"])])

    if len(rows) == 1:
        rows.append([p("-", styles["TEXT"])])

    table = Table(rows, colWidths=[CONTENT_WIDTH], repeatRows=1)
    commands = [
        ("BACKGROUND", (0, 0), (-1, -1), colors.Color(1, 1, 1, alpha=0)),
        ("BOX", (0, 0), (-1, -1), 0.8, LINE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, 0), 4),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
        ("TOPPADDING", (0, 1), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 7),
    ]

    if len(rows) >= 2:
        commands.append(("NOSPLIT", (0, 0), (-1, 1)))

    table.setStyle(TableStyle(commands))
    return table


def draw_page(canv, doc, title: str, language: str) -> None:
    width, height = A4
    labels = LANGUAGES[language]

    canv.saveState()
    try:
        if hasattr(canv, "setFillAlpha"):
            canv.setFillAlpha(0.055)
    except Exception:
        pass

    image = ImageReader(LOGO_PATH)
    image_width, image_height = image.getSize()
    scale = min(width / image_width, height / image_height) * 0.92
    draw_width = image_width * scale
    draw_height = image_height * scale

    canv.drawImage(
        image,
        (width - draw_width) / 2.0,
        (height - draw_height) / 2.0,
        width=draw_width,
        height=draw_height,
        mask="auto",
    )
    canv.restoreState()

    canv.saveState()
    canv.setFont("Roboto", 19)
    canv.setFillColor(ACCENT)
    canv.drawRightString(width - 2 * cm, height - 2.2 * cm, title)

    canv.setFont("Roboto", 8)
    canv.setFillColor(MUTED)
    footer_left = f"{DATA['website']}  |  v{DATA['version']}"
    footer_right = f"{labels['footer_page']} {canv.getPageNumber()}"
    canv.drawString(2 * cm, 1.15 * cm, footer_left)
    canv.drawRightString(width - 2 * cm, 1.15 * cm, footer_right)
    canv.restoreState()


def build_metadata_table(language: str, styles: dict) -> Table:
    labels = LANGUAGES[language]
    values = [
        f"{labels['meta_updated']}: {current_date(language)}",
        f"{labels['meta_version']}: {DATA['version']}",
        f"{labels['meta_site']}: {DATA['website']}",
    ]

    table = Table(
        [[p(value, styles["SMALL"]) for value in values]],
        colWidths=[CONTENT_WIDTH / 3.0] * 3,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def generate_pdf(language: str, output_pdf: str) -> None:
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    labels = LANGUAGES[language]
    styles = create_styles()
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    document = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.7 * cm,
        bottomMargin=1.8 * cm,
        title=labels["title"],
        author=DATA["controller"]["brand_name"],
        subject="GDPR privacy notice",
    )

    story = [
        Spacer(1, 8),
        build_metadata_table(language, styles),
        Spacer(1, 12),
        p(labels["intro"], styles["TEXT"]),
        Spacer(1, 12),
    ]

    for section in build_sections(language):
        story.append(
            box(
                section.get("title", "-"),
                section.get("paragraphs", []) or [],
                section.get("bullets", []) or [],
                styles,
            )
        )
        story.append(Spacer(1, 8))

    page_callback = partial(
        draw_page,
        title=labels["title"],
        language=language,
    )
    document.build(
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