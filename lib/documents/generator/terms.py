# lib/documents/generator/terms.py
import os
from datetime import date
from functools import partial
from html import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_PDF_SK = os.path.normpath(os.path.join(BASE_DIR, "..", "terms_sk.pdf"))
OUTPUT_PDF_EN = os.path.normpath(os.path.join(BASE_DIR, "..", "terms_en.pdf"))

# Expected project structure:
# D:\\Spectline\\lib\\documents\\generator\\terms.py
# D:\\Spectline\\public\\images\\logo_pdf.png
# D:\\Spectline\\public\\fonts\\Roboto-Regular.ttf
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


# -----------------------------------------------------------------------------
# PROVIDER DATA
# Complete the legal identity before publishing the generated documents.
# The brand name alone may not be sufficient to identify the contracting party.
# -----------------------------------------------------------------------------
DATA = {
    "version": "2.0",
    "website": "spectline.com",
    "provider": {
        "brand_name": "Spectline",
        "legal_name": "Spectline",
        "address": "Spišská Nová Ves",
        "country_sk": "Slovenská republika",
        "country_en": "Slovak Republic",
        "ico": "",
        "dic": "",
        "ic_dph": "",
        "email": "info.spectline@gmail.com",
        "phone": "",
        "registration": "",
    },
    # The percentages below are defaults only. A quotation or individual
    # agreement may set different payment and revision terms for a project.
    "commercial_defaults": {
        "deposit_percent": 50,
        "payment_due_days": 14,
        "included_revision_rounds": 2,
    },
}


LANGUAGES = {
    "sk": {
        "title": "OBCHODNÉ PODMIENKY SPECTLINE",
        "intro": (
            "Tieto obchodné podmienky upravujú objednávanie a poskytovanie kreatívnych "
            "služieb Spectline, najmä VFX, 2D a 3D animácie, 3D tvorby, motion dizajnu, "
            "strihu, postprodukcie a súvisiacich vizuálnych výstupov. Konkrétna cenová "
            "ponuka, objednávka alebo zmluva môže tieto podmienky doplniť alebo upraviť."
        ),
        "last_updated": "Posledná aktualizácia",
        "version": "Verzia",
        "website": "Web",
        "page": "Strana",
        "provider_heading": "Identifikácia poskytovateľa",
        "provider_labels": {
            "legal_name": "Obchodné meno / meno",
            "brand_name": "Obchodné označenie",
            "address": "Sídlo / adresa",
            "country": "Krajina",
            "ico": "IČO",
            "dic": "DIČ",
            "ic_dph": "IČ DPH",
            "registration": "Registrácia",
            "email": "E-mail",
            "phone": "Telefón",
            "website": "Web",
        },
    },
    "en": {
        "title": "SPECTLINE TERMS AND CONDITIONS",
        "intro": (
            "These Terms and Conditions govern the ordering and provision of Spectline's "
            "creative services, including VFX, 2D and 3D animation, 3D production, motion "
            "design, editing, post-production and related visual deliverables. A specific "
            "quotation, order or contract may supplement or amend these Terms."
        ),
        "last_updated": "Last updated",
        "version": "Version",
        "website": "Website",
        "page": "Page",
        "provider_heading": "Provider identification",
        "provider_labels": {
            "legal_name": "Legal name / name",
            "brand_name": "Trading name",
            "address": "Registered office / address",
            "country": "Country",
            "ico": "Company ID",
            "dic": "Tax ID",
            "ic_dph": "VAT ID",
            "registration": "Registration",
            "email": "Email",
            "phone": "Phone",
            "website": "Website",
        },
    },
}


ENGLISH_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def formatted_date(language: str) -> str:
    today = date.today()
    if language == "sk":
        return today.strftime("%d.%m.%Y")
    return f"{today.day} {ENGLISH_MONTHS[today.month - 1]} {today.year}"


def provider_value(key: str, language: str) -> str:
    provider = DATA["provider"]
    if key == "country":
        return provider["country_sk"] if language == "sk" else provider["country_en"]
    if key == "website":
        return DATA["website"]
    return str(provider.get(key, "")).strip()


def provider_lines(language: str) -> list[str]:
    labels = LANGUAGES[language]["provider_labels"]
    provider = DATA["provider"]
    keys = (
        "legal_name",
        "brand_name",
        "address",
        "country",
        "ico",
        "dic",
        "ic_dph",
        "registration",
        "email",
        "phone",
        "website",
    )

    lines: list[str] = []
    for key in keys:
        value = provider_value(key, language)
        if not value:
            continue

        # Do not repeat the brand if it is identical to the legal name.
        if (
            key == "brand_name"
            and value.casefold() == str(provider.get("legal_name", "")).strip().casefold()
        ):
            continue

        lines.append(f"{labels[key]}: {value}")

    return lines


def build_sections(language: str) -> list[dict]:
    email = DATA["provider"]["email"]
    website = DATA["website"]
    defaults = DATA["commercial_defaults"]
    deposit = defaults["deposit_percent"]
    due_days = defaults["payment_due_days"]
    revisions = defaults["included_revision_rounds"]

    if language == "sk":
        return [
            {
                "title": "1. Rozsah a použitie podmienok",
                "paragraphs": [
                    "Tieto obchodné podmienky sa vzťahujú na služby poskytované Spectline klientom, ktorí vystupujú ako podnikatelia, právnické osoby alebo spotrebitelia.",
                    "Súčasťou zmluvného vzťahu je najmä prijatá cenová ponuka, potvrdená objednávka, projektový brief, harmonogram, prípadná samostatná zmluva a tieto podmienky.",
                    "Ak je medzi dokumentmi rozpor, prednosť má individuálne dohodnutá zmluva alebo cenová ponuka pred týmito všeobecnými podmienkami.",
                ],
            },
            {
                "title": "2. Poskytovateľ a kontakt",
                "paragraphs": [
                    "Poskytovateľom služieb a zmluvnou stranou je subjekt identifikovaný nižšie.",
                ],
                "bullets": provider_lines("sk"),
            },
            {
                "title": "3. Definície",
                "paragraphs": ["Na účely týchto podmienok sa rozumie:"],
                "bullets": [
                    "„Poskytovateľ“ – Spectline alebo právny subjekt uvedený v časti 2.",
                    "„Klient“ – osoba alebo subjekt, ktorý si objednáva služby Poskytovateľa.",
                    "„Projekt“ – dohodnutý rozsah kreatívnych, produkčných alebo postprodukčných prác.",
                    "„Brief“ – zadanie, podklady, technické požiadavky, referencie a ciele oznámené Klientom.",
                    "„Výstup“ – finálny súbor, animácia, video, obraz, 3D model, kompozícia, grafika alebo iný výsledok Projektu určený na odovzdanie.",
                    "„Pracovné súbory“ – zdrojové, projektové, editovateľné alebo medzivýstupové súbory použité pri tvorbe.",
                    "„Revízia“ – primeraná úprava už vytvoreného návrhu v rámci pôvodne dohodnutého zadania.",
                ],
            },
            {
                "title": "4. Ponuka, objednávka a vznik zmluvy",
                "paragraphs": [
                    "Dopyt odoslaný cez web, e-mail alebo iný komunikačný kanál nie je automaticky záväznou objednávkou.",
                    "Poskytovateľ môže na základe dopytu pripraviť cenovú ponuku, odhad, harmonogram alebo návrh rozsahu prác. Ponuka platí počas lehoty uvedenej v ponuke; ak lehota uvedená nie je, platí primeraný čas vzhľadom na povahu Projektu.",
                    "Zmluva vzniká potvrdením objednávky Poskytovateľom, písomným prijatím cenovej ponuky Klientom, podpisom zmluvy alebo zaplatením dohodnutej zálohy, podľa toho, ktorá udalosť nastane ako prvá a z okolností je zrejmá vôľa strán uzavrieť zmluvu.",
                ],
                "bullets": [
                    "Poskytovateľ môže objednávku odmietnuť najmä z kapacitných, technických, právnych alebo etických dôvodov.",
                    "Ústne dohody a zmeny sú záväzné až po ich písomnom potvrdení, napríklad e-mailom.",
                ],
            },
            {
                "title": "5. Rozsah Projektu a súčinnosť Klienta",
                "paragraphs": [
                    "Rozsah Projektu vychádza z prijatej ponuky, briefu a následne potvrdených zmien. Poskytovateľ vykoná práce odborne a podľa dohodnutého kreatívneho a technického smerovania.",
                    "Klient poskytne včas úplné, pravdivé a použiteľné podklady, spätnú väzbu, schválenia, prístupy a rozhodnutia potrebné na realizáciu Projektu.",
                ],
                "bullets": [
                    "Klient zodpovedá za správnosť textov, mien, údajov, log, titulkov, prekladov a iných informácií, ktoré dodal alebo schválil.",
                    "Oneskorenie Klienta so súčinnosťou primerane posúva termíny a môže ovplyvniť cenu alebo rezervovanú produkčnú kapacitu.",
                    "Ak Klient nereaguje dlhší čas, Poskytovateľ môže Projekt pozastaviť a navrhnúť nový harmonogram.",
                ],
            },
            {
                "title": "6. Podklady Klienta a práva tretích osôb",
                "paragraphs": [
                    "Klient vyhlasuje, že má všetky potrebné práva, licencie, súhlasy a oprávnenia na použitie materiálov, ktoré poskytne Poskytovateľovi.",
                    "To zahŕňa najmä hudbu, video, fotografie, fonty, logá, ochranné známky, 3D modely, hlasové nahrávky, podobizne osôb, osobné údaje a iný chránený obsah.",
                ],
                "bullets": [
                    "Klient zodpovedá za nároky tretích osôb spôsobené použitím materiálov dodaných Klientom, pokiaľ Poskytovateľ neporušil svoje vlastné povinnosti.",
                    "Poskytovateľ môže odmietnuť použiť materiál, ktorý je zjavne nezákonný, škodlivý alebo porušuje práva tretích osôb.",
                ],
            },
            {
                "title": "7. Cena, záloha a platobné podmienky",
                "paragraphs": [
                    "Cena sa určuje individuálne podľa rozsahu, náročnosti, termínu, licencie, počtu revízií a požadovaných výstupov. Rozhodujúca je cena uvedená v prijatej ponuke alebo zmluve.",
                    f"Ak nie je dohodnuté inak, Poskytovateľ môže požadovať zálohu vo výške {deposit} % z predpokladanej ceny pred začatím prác. Štandardná lehota splatnosti faktúry je {due_days} dní, ak faktúra alebo dohoda neurčuje inú lehotu.",
                ],
                "bullets": [
                    "Ceny sú uvedené s DPH alebo bez DPH podľa údajov v ponuke a daňového postavenia Poskytovateľa.",
                    "Bankové poplatky, náklady na licencie, stock materiál, externých dodávateľov, cestovanie, expresné dodanie alebo iné mimoriadne náklady môžu byť účtované samostatne, ak boli vopred odsúhlasené alebo sú nevyhnutné na splnenie zmeneného zadania.",
                    "Poskytovateľ môže pozastaviť práce alebo odovzdanie finálnych výstupov pri omeškaní Klienta s platbou.",
                    "Zákonné úroky z omeškania a náklady spojené s uplatnením pohľadávky sa uplatnia iba v rozsahu dovolenom právnymi predpismi.",
                ],
            },
            {
                "title": "8. Termíny a harmonogram",
                "paragraphs": [
                    "Termíny sú záväzné, ak boli výslovne potvrdené ako pevné. Inak predstavujú kvalifikovaný odhad založený na informáciách dostupných pri prijatí Projektu.",
                    "Harmonogram začína plynúť po prijatí všetkých potrebných podkladov, schválení a prípadnej zálohy.",
                ],
                "bullets": [
                    "Termín sa primerane predlžuje pri oneskorenej spätnej väzbe, zmene zadania, dodatočných prácach, technických prekážkach, výpadkoch dodávateľov alebo udalostiach mimo primeranej kontroly Poskytovateľa.",
                    "Expresné alebo prioritné dodanie môže podliehať osobitnej cene a dostupnosti kapacity.",
                ],
            },
            {
                "title": "9. Návrhy, revízie a zmeny zadania",
                "paragraphs": [
                    f"Ak ponuka neurčuje inak, cena zahŕňa najviac {revisions} kolá primeraných revízií v rámci pôvodného briefu.",
                    "Klient poskytne spätnú väzbu zrozumiteľne, súhrnne a prostredníctvom dohodnutého kontaktu. Rozporuplné pokyny od viacerých osôb môžu práce pozastaviť do určenia oprávnenej schvaľujúcej osoby.",
                ],
                "bullets": [
                    "Zmena konceptu, štýlu, scenára, dĺžky, formátu, technických parametrov, už schválenej fázy alebo rozsahu sa považuje za dodatočnú prácu.",
                    "Dodatočné práce sa vykonajú po odsúhlasení vplyvu na cenu a termín; pri menších zmenách môže postačovať e-mailové potvrdenie.",
                    "Nevyužité revízie sa nepreplácajú ani neprenášajú na iný Projekt, ak nebolo dohodnuté inak.",
                ],
            },
            {
                "title": "10. Schvaľovanie a odovzdanie výstupov",
                "paragraphs": [
                    "Poskytovateľ môže odovzdávať náhľady s vodoznakom, v zníženej kvalite alebo v pracovnom formáte na účely kontroly a schválenia.",
                    "Finálne výstupy sa odovzdajú dohodnutým spôsobom a v dohodnutých formátoch po splnení platobných podmienok, ak ponuka alebo zmluva neurčuje inak.",
                ],
                "bullets": [
                    "Klient skontroluje výstupy bez zbytočného odkladu a oznámi konkrétne vady alebo nesúlad s dohodnutým zadaním.",
                    "Subjektívna zmena vkusu po schválení kreatívneho smeru nie je sama osebe vadou, môže však byť predmetom dodatočnej objednávky.",
                    "Pracovné alebo zdrojové súbory nie sú automaticky súčasťou odovzdania, pokiaľ boli výslovne zahrnuté v ponuke alebo zmluve.",
                ],
            },
            {
                "title": "11. Vady, reklamácie a oprava",
                "paragraphs": [
                    "Klient môže reklamovať preukázateľný nesúlad výstupu s dohodnutým briefom, technickými parametrami alebo záväzne schválenými požiadavkami.",
                    f"Reklamáciu je možné zaslať na {email}. Mala by obsahovať identifikáciu Projektu, popis vady, miesto jej výskytu, požadovaný spôsob nápravy a relevantné dôkazy alebo súbory.",
                ],
                "bullets": [
                    "Poskytovateľ reklamáciu posúdi a navrhne primeranú nápravu, najmä opravu, doplnenie, nové odovzdanie, primeranú zľavu alebo iný zákonný spôsob vybavenia.",
                    "Na práva spotrebiteľa sa uplatnia zákonné lehoty a pravidlá, ktoré nemožno týmito podmienkami obmedziť.",
                    "Reklamácia sa nevzťahuje na chyby spôsobené zásahom Klienta alebo tretej osoby, použitím v nepodporovanom prostredí, zmenou formátu mimo dohodnutého postupu alebo nesprávnymi podkladmi Klienta.",
                ],
            },
            {
                "title": "12. Autorské práva a licencia",
                "paragraphs": [
                    "Autorské práva a iné práva duševného vlastníctva k tvorbe Poskytovateľa zostávajú zachované v rozsahu podľa právnych predpisov. Klient nadobúda licenciu alebo iné oprávnenie na použitie výstupov iba v rozsahu dohodnutom v ponuke alebo zmluve.",
                    "Ak rozsah licencie nie je výslovne určený, vykladá sa primerane účelu Projektu, cene, povahe výstupu a spôsobu použitia, ktorý bol Poskytovateľovi pri uzatvorení zmluvy známy.",
                ],
                "bullets": [
                    "Licencia môže byť obmedzená územím, časom, médiom, účelom, kampaňou, počtom použití alebo právom na úpravu a sublicencovanie.",
                    "Licencia k finálnym výstupom vzniká po úplnom zaplatení dohodnutej ceny, ak zmluva výslovne neurčuje inak.",
                    "Pracovné postupy, know-how, šablóny, nástroje, knižnice, skripty, reusable assety a všeobecné produkčné prvky Poskytovateľa sa neprevádzajú, ak nebolo písomne dohodnuté inak.",
                    "Práva k materiálom tretích osôb sa riadia príslušnými licenčnými podmienkami a môžu byť užšie než licencia k pôvodnej tvorbe Poskytovateľa.",
                ],
            },
            {
                "title": "13. Portfólio, showreel a označenie autora",
                "paragraphs": [
                    "Poskytovateľ môže použiť verejne zverejnené finálne výstupy alebo primerané ukážky Projektu vo svojom portfóliu, showreeli, na sociálnych sieťach, súťažiach a pri prezentácii svojich služieb, ak to bolo dohodnuté, vyplýva to z okolností spolupráce alebo na to existuje iný právny základ.",
                    "Ak je Projekt dôverný, podlieha embargu alebo Klient použitie v portfóliu odmietne pred uzatvorením zmluvy, strany si dohodnú obmedzenie alebo vylúčenie takého použitia.",
                ],
                "bullets": [
                    "Poskytovateľ nebude bez oprávnenia zverejňovať neverejné obchodné informácie, osobné údaje ani materiály, na ktoré nemá potrebné práva.",
                    "Prípadné uvedenie mena alebo značky Klienta slúži iba na identifikáciu realizovaného Projektu a neznamená partnerstvo ani podporu iných služieb.",
                ],
            },
            {
                "title": "14. Externí dodávatelia, stock materiály a technológie",
                "paragraphs": [
                    "Poskytovateľ môže pri realizácii Projektu zapojiť dôveryhodných externých spolupracovníkov alebo dodávateľov, pričom zostáva zodpovedný za riadenie svojho plnenia v dohodnutom rozsahu.",
                    "Projekt môže využívať licencované fonty, hudbu, stock obsah, pluginy, cloudové služby, render farmy alebo iné technológie tretích strán.",
                ],
                "bullets": [
                    "Použitie materiálov tretích strán môže podliehať osobitným licenčným, územným alebo časovým obmedzeniam.",
                    "Ak je na Projekt vhodné použiť generatívne alebo iné AI nástroje, Poskytovateľ zohľadní dohodnuté požiadavky na dôvernosť, licencie a technické obmedzenia. Klient môže požadovať, aby sa AI nástroje nepoužili; taká požiadavka môže ovplyvniť cenu alebo harmonogram.",
                ],
            },
            {
                "title": "15. Dôvernosť a projektové súbory",
                "paragraphs": [
                    "Strany budú chrániť dôverné informácie získané počas spolupráce a použijú ich iba na účely Projektu, plnenia právnych povinností alebo ochrany svojich práv.",
                    "Za dôverné sa nepovažujú informácie, ktoré boli verejne známe bez porušenia povinnosti, boli získané zákonným spôsobom od tretej osoby alebo musia byť sprístupnené podľa zákona.",
                ],
                "bullets": [
                    "Poskytovateľ nie je povinný uchovávať pracovné a projektové súbory neobmedzene. Klient si po odovzdaní zabezpečí vlastnú zálohu finálnych výstupov.",
                    "Dlhodobá archivácia, opätovné otvorenie starého Projektu alebo migrácia pracovných súborov môže byť spoplatnená a závisí od technickej dostupnosti.",
                ],
            },
            {
                "title": "16. Pozastavenie, zrušenie a ukončenie Projektu",
                "paragraphs": [
                    "Klient môže požiadať o pozastavenie alebo zrušenie Projektu. Poskytovateľ má nárok na úhradu riadne vykonaných prác, schválených nákladov, objednaných licencií a primeraných nevratných záväzkov vzniknutých do dátumu ukončenia.",
                    "Výška storno poplatku alebo spôsob vyúčtovania môže byť uvedený v cenovej ponuke alebo zmluve. Ak uvedený nie je, vyúčtuje sa preukázateľne vykonaná práca a nevratné náklady.",
                ],
                "bullets": [
                    "Poskytovateľ môže Projekt pozastaviť alebo ukončiť pri podstatnom porušení povinností Klienta, omeškaní s platbou, dlhodobej neposkytnutej súčinnosti, nezákonnom zadaní alebo ohrození bezpečnosti.",
                    "Pred ukončením pre napraviteľné porušenie poskytne Poskytovateľ primeranú lehotu na nápravu, ak to okolnosti umožňujú.",
                    "Po ukončení zmluvy sa práva na použitie rozpracovaných alebo nezaplatených výstupov neposkytujú, ak nebolo písomne dohodnuté inak.",
                ],
            },
            {
                "title": "17. Zodpovednosť a vyššia moc",
                "paragraphs": [
                    "Každá strana zodpovedá za škodu spôsobenú porušením svojich povinností v rozsahu stanovenom právnymi predpismi a zmluvou.",
                    "Poskytovateľ nezodpovedá za omeškanie alebo nemožnosť plnenia spôsobenú udalosťami mimo jeho primeranej kontroly, napríklad rozsiahlym výpadkom infraštruktúry, živelnou udalosťou, zásahom verejnej moci, vojnou, nepokojmi, epidémiou alebo závažným zlyhaním dodávateľa, ak prijal primerané opatrenia na zmiernenie následkov.",
                ],
                "bullets": [
                    "Poskytovateľ nezodpovedá za výsledky kampane, obchodný úspech, sledovanosť alebo výkonnosť, pokiaľ nebola konkrétna merateľná záruka výslovne dohodnutá.",
                    "Žiadne ustanovenie nevylučuje zodpovednosť, ktorú podľa kogentného práva nemožno vylúčiť alebo obmedziť, ani zákonné práva spotrebiteľa.",
                ],
            },
            {
                "title": "18. Spotrebiteľ a odstúpenie od zmluvy na diaľku",
                "paragraphs": [
                    "Ak je Klient spotrebiteľom a zmluva bola uzavretá na diaľku alebo mimo prevádzkových priestorov Poskytovateľa, môže mať právo odstúpiť od zmluvy bez uvedenia dôvodu v zákonnej lehote, spravidla do 14 dní od uzavretia zmluvy o službe.",
                    "Ak spotrebiteľ výslovne požiada o začatie poskytovania služby pred uplynutím lehoty na odstúpenie, pri odstúpení môže byť povinný zaplatiť pomernú časť ceny za plnenie poskytnuté do doručenia odstúpenia. Po úplnom poskytnutí služby môže právo na odstúpenie zaniknúť, ak boli splnené zákonné podmienky vrátane predchádzajúceho výslovného súhlasu a poučenia spotrebiteľa.",
                ],
                "bullets": [
                    f"Oznámenie o odstúpení možno zaslať na {email} alebo iným jednoznačným vyhlásením doručeným Poskytovateľovi.",
                    "Toto ustanovenie sa neuplatní na podnikateľov a neobmedzuje žiadne právo spotrebiteľa, ktoré mu priznáva kogentný právny predpis.",
                ],
            },
            {
                "title": "19. Ochrana osobných údajov",
                "paragraphs": [
                    f"Spracúvanie osobných údajov sa riadi samostatným dokumentom Ochrana osobných údajov (GDPR), ktorý je dostupný na webe {website}.",
                    "Klient zabezpečí, že osobné údaje obsiahnuté v projektových podkladoch poskytuje a používa zákonným spôsobom. Ak Poskytovateľ spracúva osobné údaje v mene Klienta ako sprostredkovateľ, strany podľa potreby uzatvoria príslušnú dohodu o spracúvaní údajov.",
                ],
            },
            {
                "title": "20. Sťažnosti a alternatívne riešenie spotrebiteľských sporov",
                "paragraphs": [
                    f"Klient môže otázku, sťažnosť alebo žiadosť o nápravu zaslať na {email}. Strany sa pokúsia spor vyriešiť prednostne dohodou.",
                    "Spotrebiteľ, ktorý nie je spokojný so spôsobom vybavenia žiadosti o nápravu alebo na ňu nedostal odpoveď v zákonnej lehote, sa môže obrátiť na príslušný subjekt alternatívneho riešenia spotrebiteľských sporov podľa platných právnych predpisov, najmä na Slovenskú obchodnú inšpekciu, ak je vec v jej pôsobnosti.",
                ],
            },
            {
                "title": "21. Rozhodné právo a súdy",
                "paragraphs": [
                    "Zmluvný vzťah sa riadi právnym poriadkom Slovenskej republiky, pokiaľ kogentné pravidlá medzinárodného práva súkromného neurčujú inak.",
                    "Voľba slovenského práva nezbavuje spotrebiteľa ochrany, ktorú mu poskytujú kogentné ustanovenia práva štátu jeho obvyklého pobytu, ak sa také ustanovenia uplatnia.",
                    "Ak sa spor nepodarí vyriešiť dohodou, rozhodne ho vecne a miestne príslušný súd podľa platných právnych predpisov.",
                ],
            },
            {
                "title": "22. Zmeny a záverečné ustanovenia",
                "paragraphs": [
                    "Poskytovateľ môže tieto podmienky primerane aktualizovať, najmä pri zmene služieb, procesov alebo právnych predpisov. Na konkrétny Projekt sa spravidla použije verzia účinná pri vzniku zmluvy, ak sa strany nedohodnú inak alebo zmena nevyplýva zo zákona.",
                    "Ak je niektoré ustanovenie neplatné alebo nevykonateľné, ostatné ustanovenia zostávajú v platnosti. Neplatné ustanovenie sa nahradí zákonným riešením, ktoré sa čo najviac približuje jeho hospodárskemu účelu.",
                    "Anglické znenie je prekladom slovenského znenia. Pri rozdieloch vo výklade má prednosť slovenské znenie, pokiaľ kogentné právo nevyžaduje inak.",
                    f"Posledná aktualizácia: {formatted_date('sk')}",
                    f"Verzia: {DATA['version']}",
                ],
            },
        ]

    return [
        {
            "title": "1. Scope and application",
            "paragraphs": [
                "These Terms apply to services provided by Spectline to clients acting as businesses, legal entities or consumers.",
                "The contractual relationship consists primarily of the accepted quotation, confirmed order, project brief, schedule, any separate agreement and these Terms.",
                "If the documents conflict, an individually agreed contract or quotation takes precedence over these general Terms.",
            ],
        },
        {
            "title": "2. Provider and contact details",
            "paragraphs": [
                "The services are provided and the contract is entered into by the entity identified below.",
            ],
            "bullets": provider_lines("en"),
        },
        {
            "title": "3. Definitions",
            "paragraphs": ["For the purposes of these Terms:"],
            "bullets": [
                '"Provider" means Spectline or the legal entity identified in Section 2.',
                '"Client" means the person or entity ordering the Provider\'s services.',
                '"Project" means the agreed scope of creative, production or post-production work.',
                '"Brief" means the instructions, materials, technical requirements, references and objectives supplied by the Client.',
                '"Deliverable" means a final file, animation, video, image, 3D model, composition, graphic or other Project result intended for delivery.',
                '"Working Files" means source, project, editable or intermediate files used during production.',
                '"Revision" means a reasonable adjustment to an existing proposal within the originally agreed Brief.',
            ],
        },
        {
            "title": "4. Quotation, order and formation of contract",
            "paragraphs": [
                "An enquiry submitted through the website, by email or through another channel is not automatically a binding order.",
                "Based on an enquiry, the Provider may prepare a quotation, estimate, schedule or proposed scope. A quotation remains valid for the period stated in it; where no period is stated, it remains valid for a reasonable period considering the nature of the Project.",
                "A contract is formed when the Provider confirms the order, the Client accepts the quotation in writing, the parties sign a contract or the Client pays the agreed deposit, whichever occurs first and clearly demonstrates the parties' intention to contract.",
            ],
            "bullets": [
                "The Provider may decline an order for capacity, technical, legal or ethical reasons.",
                "Oral agreements and changes become binding only after written confirmation, including confirmation by email.",
            ],
        },
        {
            "title": "5. Project scope and Client cooperation",
            "paragraphs": [
                "The Project scope is based on the accepted quotation, Brief and subsequently confirmed changes. The Provider will perform the work professionally and in accordance with the agreed creative and technical direction.",
                "The Client will provide complete, accurate and usable materials, feedback, approvals, access and decisions required to perform the Project on time.",
            ],
            "bullets": [
                "The Client is responsible for the accuracy of text, names, data, logos, captions, translations and other information supplied or approved by the Client.",
                "A delay in Client cooperation will reasonably extend deadlines and may affect the price or reserved production capacity.",
                "If the Client remains unresponsive for an extended period, the Provider may suspend the Project and propose a revised schedule.",
            ],
        },
        {
            "title": "6. Client materials and third-party rights",
            "paragraphs": [
                "The Client represents that it holds all rights, licences, consents and permissions required for materials supplied to the Provider.",
                "This includes music, video, photographs, fonts, logos, trademarks, 3D models, voice recordings, likenesses, personal data and other protected content.",
            ],
            "bullets": [
                "The Client is responsible for third-party claims arising from Client-supplied materials, except to the extent caused by the Provider's own breach.",
                "The Provider may refuse to use material that is manifestly unlawful, harmful or infringes third-party rights.",
            ],
        },
        {
            "title": "7. Price, deposit and payment",
            "paragraphs": [
                "Pricing is determined individually according to scope, complexity, timing, licence, number of revisions and required deliverables. The price stated in the accepted quotation or contract controls.",
                f"Unless agreed otherwise, the Provider may require a deposit of {deposit}% of the estimated price before work begins. The standard invoice due date is {due_days} days unless the invoice or agreement specifies a different period.",
            ],
            "bullets": [
                "Prices are stated inclusive or exclusive of VAT as indicated in the quotation and according to the Provider's tax status.",
                "Bank fees, licence fees, stock assets, external suppliers, travel, rush delivery or other exceptional costs may be charged separately if approved in advance or required by a changed Brief.",
                "The Provider may suspend work or final delivery while the Client is overdue with payment.",
                "Statutory default interest and recovery costs apply only to the extent permitted by law.",
            ],
        },
        {
            "title": "8. Deadlines and schedule",
            "paragraphs": [
                "Deadlines are binding when expressly confirmed as fixed. Otherwise, they are good-faith estimates based on information available when the Project is accepted.",
                "The schedule begins once all required materials, approvals and any deposit have been received.",
            ],
            "bullets": [
                "Deadlines are reasonably extended by delayed feedback, changes in scope, additional work, technical obstacles, supplier outages or events outside the Provider's reasonable control.",
                "Rush or priority delivery may be subject to additional pricing and production availability.",
            ],
        },
        {
            "title": "9. Concepts, revisions and scope changes",
            "paragraphs": [
                f"Unless the quotation states otherwise, the price includes up to {revisions} rounds of reasonable revisions within the original Brief.",
                "The Client will provide clear, consolidated feedback through the agreed contact. Conflicting instructions from multiple persons may suspend work until an authorised approver is identified.",
            ],
            "bullets": [
                "A change to the concept, style, script, duration, format, technical parameters, an already approved stage or the overall scope constitutes additional work.",
                "Additional work will proceed after the impact on price and timing is approved; email confirmation may be sufficient for minor changes.",
                "Unused revisions are not refundable and do not transfer to another Project unless agreed otherwise.",
            ],
        },
        {
            "title": "10. Approval and delivery",
            "paragraphs": [
                "The Provider may supply watermarked, reduced-quality or working-format previews for review and approval.",
                "Final Deliverables will be supplied using the agreed method and formats after payment conditions have been satisfied, unless the quotation or contract provides otherwise.",
            ],
            "bullets": [
                "The Client will review Deliverables without undue delay and identify any specific defect or departure from the agreed Brief.",
                "A subjective change in taste after approval of the creative direction is not by itself a defect, but may be commissioned as additional work.",
                "Working Files or source files are not automatically included unless expressly listed in the quotation or contract.",
            ],
        },
        {
            "title": "11. Defects, complaints and remedies",
            "paragraphs": [
                "The Client may report a demonstrable failure of a Deliverable to match the agreed Brief, technical specifications or binding approved requirements.",
                f"A complaint may be sent to {email}. It should identify the Project, describe the defect and where it occurs, state the requested remedy and include relevant evidence or files.",
            ],
            "bullets": [
                "The Provider will assess the complaint and propose an appropriate remedy, such as correction, completion, redelivery, a proportionate price reduction or another remedy required by law.",
                "Statutory consumer deadlines and rights apply and are not restricted by these Terms.",
                "A complaint does not cover errors caused by Client or third-party intervention, use in an unsupported environment, format conversion outside the agreed workflow or incorrect Client materials.",
            ],
        },
        {
            "title": "12. Copyright and licence",
            "paragraphs": [
                "Copyright and other intellectual property rights in the Provider's work remain protected to the extent provided by law. The Client receives a licence or other right to use the Deliverables only within the scope agreed in the quotation or contract.",
                "Where the licence scope is not expressly stated, it will be interpreted reasonably according to the Project's purpose, price, nature of the Deliverable and the intended use known to the Provider when the contract was formed.",
            ],
            "bullets": [
                "A licence may be limited by territory, duration, medium, purpose, campaign, number of uses or the right to modify or sublicense.",
                "The licence to final Deliverables takes effect after full payment unless the contract expressly provides otherwise.",
                "The Provider's workflows, know-how, templates, tools, libraries, scripts, reusable assets and general production components are not transferred unless agreed in writing.",
                "Rights in third-party materials are governed by their applicable licence terms and may be narrower than the licence to the Provider's original work.",
            ],
        },
        {
            "title": "13. Portfolio, showreel and credit",
            "paragraphs": [
                "The Provider may use publicly released final Deliverables or reasonable Project excerpts in its portfolio, showreel, social media, awards submissions and service presentations where this was agreed, follows from the circumstances of the engagement or is supported by another valid legal basis.",
                "If the Project is confidential, embargoed or the Client objects to portfolio use before the contract is formed, the parties will agree an appropriate restriction or exclusion.",
            ],
            "bullets": [
                "The Provider will not disclose non-public business information, personal data or material it is not authorised to publish.",
                "Any reference to the Client's name or brand identifies the completed Project only and does not imply a partnership or endorsement of other services.",
            ],
        },
        {
            "title": "14. Subcontractors, stock assets and technology",
            "paragraphs": [
                "The Provider may engage trusted subcontractors or suppliers while remaining responsible for managing its contractual performance within the agreed scope.",
                "A Project may use licensed fonts, music, stock content, plugins, cloud services, render farms or other third-party technology.",
            ],
            "bullets": [
                "Third-party materials may be subject to separate licence, territory or time restrictions.",
                "Where generative or other AI tools are appropriate for a Project, the Provider will take account of agreed confidentiality, licensing and technical requirements. The Client may request that AI tools not be used; this may affect price or schedule.",
            ],
        },
        {
            "title": "15. Confidentiality and Project files",
            "paragraphs": [
                "The parties will protect confidential information obtained during the engagement and use it only for the Project, legal compliance or the protection of their rights.",
                "Information is not confidential where it became public without breach, was lawfully obtained from a third party or must be disclosed by law.",
            ],
            "bullets": [
                "The Provider is not required to retain Working Files or Project files indefinitely. The Client will maintain its own backup of final Deliverables after delivery.",
                "Long-term archiving, reopening an old Project or migrating Working Files may be chargeable and depends on technical availability.",
            ],
        },
        {
            "title": "16. Suspension, cancellation and termination",
            "paragraphs": [
                "The Client may request suspension or cancellation of a Project. The Provider is entitled to payment for properly completed work, approved costs, purchased licences and reasonable non-refundable commitments incurred up to termination.",
                "A cancellation fee or settlement method may be stated in the quotation or contract. Where none is stated, documented work performed and non-refundable costs will be invoiced.",
            ],
            "bullets": [
                "The Provider may suspend or terminate a Project for a material Client breach, overdue payment, prolonged failure to cooperate, unlawful instructions or a security risk.",
                "For a remediable breach, the Provider will give a reasonable opportunity to remedy it where circumstances permit.",
                "Termination does not grant rights to use unfinished or unpaid Deliverables unless agreed in writing.",
            ],
        },
        {
            "title": "17. Liability and force majeure",
            "paragraphs": [
                "Each party is responsible for loss caused by its breach to the extent provided by applicable law and the contract.",
                "The Provider is not liable for delay or non-performance caused by events outside its reasonable control, including major infrastructure outages, natural disasters, government action, war, civil unrest, epidemic or serious supplier failure, provided reasonable mitigation steps are taken.",
            ],
            "bullets": [
                "The Provider does not guarantee campaign performance, commercial success, audience figures or other business outcomes unless a specific measurable guarantee was expressly agreed.",
                "Nothing excludes liability that cannot lawfully be excluded or limited, or any mandatory consumer right.",
            ],
        },
        {
            "title": "18. Consumers and withdrawal from distance contracts",
            "paragraphs": [
                "Where the Client is a consumer and the contract was concluded at a distance or away from the Provider's business premises, the Client may have a statutory right to withdraw without giving a reason, generally within 14 days after the service contract is concluded.",
                "If the consumer expressly requests performance to begin before the withdrawal period expires, the consumer may have to pay a proportionate amount for services supplied before withdrawal. Once the service has been fully performed, the right of withdrawal may be lost where the statutory conditions, including prior express consent and acknowledgement, have been met.",
            ],
            "bullets": [
                f"A withdrawal notice may be sent to {email} or made by another clear statement delivered to the Provider.",
                "This Section does not apply to business Clients and does not limit any mandatory consumer right.",
            ],
        },
        {
            "title": "19. Personal data protection",
            "paragraphs": [
                f"Personal data processing is governed by the separate Personal Data Protection (GDPR) notice available at {website}.",
                "The Client will ensure that personal data contained in Project materials is supplied and used lawfully. Where the Provider processes personal data on the Client's behalf as a processor, the parties will enter into an appropriate data processing agreement where required.",
            ],
        },
        {
            "title": "20. Complaints and alternative dispute resolution",
            "paragraphs": [
                f"The Client may send a question, complaint or request for redress to {email}. The parties will first seek to resolve disputes amicably.",
                "A consumer dissatisfied with the response to a request for redress, or who did not receive a response within the statutory period, may apply to a competent alternative dispute resolution entity under applicable law, including the Slovak Trade Inspection where the matter falls within its authority.",
            ],
        },
        {
            "title": "21. Governing law and courts",
            "paragraphs": [
                "The contractual relationship is governed by the laws of the Slovak Republic unless mandatory conflict-of-law rules provide otherwise.",
                "The choice of Slovak law does not deprive a consumer of mandatory protections available under the law of the consumer's habitual residence where those protections apply.",
                "If a dispute cannot be resolved amicably, it will be decided by the court having jurisdiction under applicable law.",
            ],
        },
        {
            "title": "22. Amendments and final provisions",
            "paragraphs": [
                "The Provider may reasonably update these Terms, particularly following changes to services, processes or law. A Project will normally remain subject to the version effective when the contract was formed, unless the parties agree otherwise or a change is required by law.",
                "If any provision is invalid or unenforceable, the remaining provisions remain effective. The invalid provision will be replaced by a lawful solution that most closely reflects its commercial purpose.",
                "The English version is a translation of the Slovak version. If interpretations differ, the Slovak version prevails unless mandatory law requires otherwise.",
                f"Last updated: {formatted_date('en')}",
                f"Version: {DATA['version']}",
            ],
        },
    ]


def ensure_paths() -> None:
    if not os.path.isfile(FONT_PATH):
        raise FileNotFoundError(
            f"Missing font: {FONT_PATH}\n"
            "Expected location: <project>/public/fonts/Roboto-Regular.ttf"
        )
    if not os.path.isfile(LOGO_PATH):
        raise FileNotFoundError(
            f"Missing logo: {LOGO_PATH}\n"
            "Expected location: <project>/public/images/logo_pdf.png"
        )


def register_fonts() -> None:
    if "Roboto" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("Roboto", FONT_PATH))


def safe_text(value: object) -> str:
    return escape(str(value), quote=False).replace("\n", "<br/>")


def p(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(safe_text(text), style)


def create_styles() -> dict[str, ParagraphStyle]:
    return {
        "TEXT": ParagraphStyle(
            "TEXT",
            fontName="Roboto",
            fontSize=9.4,
            leading=12.6,
            textColor=INK,
            spaceBefore=0,
            spaceAfter=0,
        ),
        "SMALL": ParagraphStyle(
            "SMALL",
            fontName="Roboto",
            fontSize=8.7,
            leading=11.5,
            textColor=MUTED,
            spaceBefore=0,
            spaceAfter=0,
        ),
        "BOX_HEAD": ParagraphStyle(
            "BOX_HEAD",
            fontName="Roboto",
            fontSize=9,
            leading=12,
            textColor=ACCENT_DARK,
            spaceBefore=0,
            spaceAfter=0,
        ),
    }


def box(title: str, paragraphs: list[str], bullets: list[str], styles: dict) -> Table:
    rows: list[list[Paragraph]] = [[p(title, styles["BOX_HEAD"])]]

    for paragraph in paragraphs:
        if str(paragraph).strip():
            rows.append([p(paragraph, styles["TEXT"])])

    for bullet in bullets:
        if str(bullet).strip():
            rows.append([p(f"• {bullet}", styles["TEXT"])])

    if len(rows) == 1:
        rows.append([p("—", styles["TEXT"])])

    table = Table(rows, colWidths=[CONTENT_WIDTH], repeatRows=1, hAlign="LEFT")

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

    # Keep each section heading with at least its first body row.
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
            canv.setFillAlpha(0.06)
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
    canv.drawString(2 * cm, 1.2 * cm, f"{DATA['website']}  |  {formatted_date(language)}")
    canv.drawRightString(
        width - 2 * cm,
        1.2 * cm,
        f"{labels['page']} {doc.page}",
    )

    canv.setTitle(title)
    canv.setAuthor(DATA["provider"]["brand_name"])
    canv.setSubject("Terms and Conditions" if language == "en" else "Obchodné podmienky")
    canv.restoreState()


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
        bottomMargin=2 * cm,
        title=labels["title"],
        author=DATA["provider"]["brand_name"],
        subject="Terms and Conditions" if language == "en" else "Obchodné podmienky",
    )

    story = [
        Spacer(1, 12),
        p(labels["intro"], styles["TEXT"]),
        Spacer(1, 7),
        p(
            f"{labels['last_updated']}: {formatted_date(language)}  |  "
            f"{labels['version']}: {DATA['version']}  |  "
            f"{labels['website']}: {DATA['website']}",
            styles["SMALL"],
        ),
        Spacer(1, 10),
    ]

    for section in build_sections(language):
        story.append(
            box(
                section.get("title", "—"),
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