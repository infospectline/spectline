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
#   images/logo_black.png
#   fonts/Roboto-Regular.ttf
LOGO_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "..", "..", "public", "images", "logo_black.png")
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
            "Spectline poskytuje softvérové riešenie založené na umelej inteligencii "
            "na analýzu obchodnej histórie, identifikáciu obchodných vzorcov a podporu "
            "alebo automatizáciu obchodovania prostredníctvom kompatibilných platforiem, "
            "vrátane MetaTrader 4 a MetaTrader 5 podľa dostupných funkcií. Toto oznámenie "
            "vysvetľuje, aké osobné údaje spracúvame pri návšteve webu, registrácii a "
            "používaní účtu a softvéru, pri komunikácii a odbere noviniek, na aké účely "
            "údaje používame, ako dlho ich uchovávame a aké práva máte podľa GDPR."
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
            "Spectline provides AI-based software for analysing trading history, "
            "identifying trading patterns and supporting or automating trading through "
            "compatible platforms, including MetaTrader 4 and MetaTrader 5 where the "
            "relevant features are available. This notice explains what personal data "
            "we process when you visit our website, register for and use an account or "
            "the software, contact us or subscribe to updates, why we use the data, "
            "how long we retain it and what rights you have under the GDPR."
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
                "title": "1. Prevádzkovateľ a kontaktné údaje",
                "paragraphs": [
                    "Prevádzkovateľom osobných údajov je subjekt identifikovaný nižšie. Prevádzkovateľ určuje účely a prostriedky spracúvania osobných údajov. Ak máte otázky týkajúce sa ochrany osobných údajov alebo chcete uplatniť svoje práva, kontaktujte nás na uvedenej e-mailovej adrese.",
                ],
                "bullets": controller_lines("sk"),
            },
            {
                "title": "2. Rozsah oznámenia a zdroje údajov",
                "paragraphs": [
                    "Toto oznámenie sa vzťahuje na spracúvanie osobných údajov pri návšteve webovej stránky, vytvorení a správe používateľského účtu, používaní softvéru Spectline, využívaní jeho analytických a automatizačných funkcií, komunikácii s nami a odbere produktových noviniek.",
                    "Údaje získavame najmä priamo od vás pri registrácii, úprave účtu, zadávaní údajov do softvéru, kontaktovaní podpory alebo prihlásení na odber noviniek. Ďalej môžu vznikať pri používaní webu a softvéru ako technické a prevádzkové záznamy. Údaje z prepojenej obchodnej platformy alebo brokerského prostredia získavame iba v rozsahu, v akom ich konkrétna funkcia skutočne sprístupňuje a vy ste dané prepojenie aktivovali alebo autorizovali.",
                ],
            },
            {
                "title": "3. Kategórie osobných údajov",
                "paragraphs": [
                    "Rozsah spracúvaných údajov závisí od toho, či iba navštívite web, vytvoríte si účet, používate obchodné a AI funkcie alebo nás kontaktujete. Spracúvame iba údaje primerané príslušnému účelu.",
                ],
                "bullets": [
                    "Identifikačné a kontaktné údaje: meno, priezvisko, e-mail, adresa a mesto; podľa toho, čo vyplníte alebo čo je potrebné pre účet, aj telefónne číslo, názov spoločnosti a webová stránka.",
                    "Údaje o účte a používaní služby: identifikátor účtu, stav registrácie a schválenia, nastavenia účtu, prihlasovacie a bezpečnostné záznamy, história prihlásení a požiadavky na podporu.",
                    "Obchodné a trhové údaje, ktoré do služby nahráte, zadáte alebo sprístupníte prostredníctvom aktivovaného prepojenia: história obchodov a pokynov, obchodované nástroje, časové rámce, vstupné a výstupné ceny, objemy, Stop Loss, Take Profit, časové údaje, výsledky obchodov a identifikátory obchodného účtu alebo platformy.",
                    "Údaje o stratégii a používaní AI funkcií: obchodné setupy, pravidlá a parametre stratégie, nastavenia bota, údaje použité na tréning alebo spätné testovanie, analytické výstupy, rozpoznané vzorce, štatistiky, signály a záznamy o vykonaní funkcií softvéru.",
                    "Technické a prevádzkové údaje: IP adresa, typ zariadenia, operačný systém, prehliadač, identifikátory relácie, čas prístupu, chybové hlásenia, bezpečnostné a systémové logy a údaje o používaní webu.",
                    "Zmluvné, platobné a účtovné údaje: objednávky, fakturačné údaje, história platieb, údaje o predplatnom a komunikácia súvisiaca s platbou. Údaje o platobnej karte spracúva spravidla príslušný poskytovateľ platobnej služby, ak je použitý.",
                    "Komunikačné a marketingové údaje: obsah správ zaslaných cez kontaktný formulár alebo e-mail, e-mailová adresa použitá na odber noviniek a záznamy o udelení, zmene alebo odvolaní súhlasu.",
                ],
            },
            {
                "title": "4. Účely a právne základy spracúvania",
                "paragraphs": [
                    "Osobné údaje spracúvame v súlade s článkom 6 GDPR a vždy na konkrétny, vopred určený účel. Podľa okolností ide najmä o tieto účely a právne základy:",
                ],
                "bullets": [
                    "Vytvorenie a správa účtu, overenie používateľa, vybavenie registrácie a sprístupnenie funkcií služby - plnenie zmluvy alebo vykonanie opatrení pred uzatvorením zmluvy podľa článku 6 ods. 1 písm. b) GDPR.",
                    "Poskytovanie AI analýzy obchodnej histórie, identifikácia vzorcov, tréning používateľom požadovanej stratégie, spätné testovanie a poskytovanie ďalších funkcií softvéru - plnenie zmluvy podľa článku 6 ods. 1 písm. b) GDPR.",
                    "Spracovanie objednávok, predplatného, platieb, fakturácie, účtovníctva a plnenie daňových alebo iných zákonných povinností - plnenie zmluvy a plnenie zákonnej povinnosti podľa článku 6 ods. 1 písm. b) a c) GDPR.",
                    "Prevádzka, údržba, diagnostika chýb, zabezpečenie účtov a systémov, predchádzanie podvodom a zneužitiu a ochrana právnych nárokov - oprávnený záujem podľa článku 6 ods. 1 písm. f) GDPR, po posúdení jeho primeranosti a vašich práv.",
                    "Odpovedanie na všeobecné otázky a vybavovanie dopytov, ktoré predchádzajú zmluve - vykonanie opatrení pred uzatvorením zmluvy alebo oprávnený záujem na riadnej obchodnej komunikácii podľa okolností.",
                    "Zasielanie dobrovoľných produktových noviniek a používanie nepovinných analytických alebo marketingových cookies - váš súhlas podľa článku 6 ods. 1 písm. a) GDPR; súhlas môžete kedykoľvek odvolať.",
                ],
            },
            {
                "title": "5. Používateľský účet a poskytovanie softvéru",
                "paragraphs": [
                    "Pri registrácii a správe účtu spracúvame údaje potrebné na vytvorenie účtu, komunikáciu s vami, správu prístupových oprávnení, prípadné administrátorské schválenie registrácie, poskytovanie zákazníckej podpory a plnenie zmluvy. V aktuálnom registračnom procese sú potrebné najmä meno, priezvisko, e-mail, adresa, mesto a prihlasovacie údaje; bez povinných údajov nemusí byť možné účet vytvoriť alebo službu používať. Telefón, názov spoločnosti a webová stránka sa poskytujú iba vtedy, ak ich formulár vyžaduje alebo ich sami doplníte. Údaje označené ako nepovinné môžete neposkytnúť.",
                    "Údaje zadané do účtu používame aj na uloženie vašich nastavení a zachovanie kontinuity používania softvéru. Ak účet zrušíte, spracúvanie ukončíme alebo obmedzíme podľa účelu, zákonných povinností, pravidiel uchovávania a potreby ochrany právnych nárokov.",
                ],
            },
            {
                "title": "6. Obchodná história, AI analýza a automatizované obchodovanie",
                "paragraphs": [
                    "Ak používate obchodné funkcie Spectline, údaje o vašej obchodnej histórii a nastaveniach spracúvame na analýzu obchodných vzorcov, tréning alebo konfiguráciu stratégie určenej pre vaše používanie, spätné testovanie, vyhodnocovanie výsledkov a poskytovanie výstupov softvéru. Konkrétny rozsah závisí od funkcií, ktoré aktivujete, a od údajov, ktoré do služby zadáte alebo sprístupníte.",
                    "Ak aktivujete automatizované obchodovanie a autorizujete prepojenie s kompatibilnou platformou, softvér môže podľa vašich nastavení vytvárať alebo odovzdávať obchodné pokyny do príslušného prostredia. Takéto spracúvanie môže mať finančné dôsledky. Obchodná história a výstupy AI sa používajú na poskytovanie funkcií služby; ich použitie na iný, nezlučiteľný účel alebo na tréning všeobecného modelu mimo vášho účtu by vyžadovalo samostatne posúdený právny základ a príslušné informovanie.",
                    "MetaTrader 4, MetaTrader 5 a broker, u ktorého máte obchodný účet, môžu spracúvať údaje aj na vlastné účely a podľa vlastných podmienok ochrany osobných údajov. To, či a aké údaje sa medzi Spectline, platformou a brokerom prenášajú, závisí od konkrétnej technickej integrácie. Údaje neposkytujeme týmto subjektom automaticky iba preto, že ich produkty spomíname na webe.",
                ],
            },
            {
                "title": "7. Údaje tretích osôb a bezpečné používanie",
                "paragraphs": [
                    "Do softvéru zadávajte iba údaje, ktoré ste oprávnení používať a sprístupniť. Ak obchodné alebo iné nahrané súbory obsahujú osobné údaje ďalších osôb, ste povinní zabezpečiť právny základ ich poskytnutia a splnenie príslušných informačných povinností. Ak Spectline spracúva osobné údaje iných osôb výlučne podľa vašich pokynov ako sprostredkovateľ, podmienky takého spracúvania sa podľa potreby upravia samostatnou dohodou podľa článku 28 GDPR.",
                    "Do kontaktného formulára ani bežnej e-mailovej komunikácie neposielajte heslá k brokerským účtom, privátne kľúče ani iné tajné autentifikačné údaje. Ak konkrétna integračná funkcia vyžaduje autorizačný token alebo iný prístupový údaj, spracúvanie musí zodpovedať technickému spôsobu integrácie a bezpečnostným informáciám poskytnutým pri jej aktivácii.",
                    "Služba nie je určená na spracúvanie osobitných kategórií osobných údajov podľa článku 9 GDPR. Takéto údaje do služby nenahrávajte, pokiaľ na to neexistuje osobitný právny dôvod a vopred dohodnutý postup.",
                ],
            },
            {
                "title": "8. Kontaktný formulár a dopyty",
                "paragraphs": [
                    "Ak nás kontaktujete cez formulár alebo e-mail, spracúvame údaje, ktoré nám poskytnete, najmä meno, e-mail, tému a obsah správy. Používame ich na odpoveď, vybavenie požiadavky a prípadné kroky pred uzatvorením zmluvy.",
                    "Ak dopyt nevedie k zmluvnej spolupráci, komunikáciu spravidla uchovávame najviac {months} mesiacov od posledného kontaktu, pokiaľ dlhšie uchovanie nie je potrebné na ochranu právnych nárokov alebo z iného zákonného dôvodu.".format(months=inquiry_months),
                ],
            },
            {
                "title": "9. Cookies a technické logy",
                "paragraphs": [
                    "Web môže používať technické cookies alebo podobné technológie, ktoré sú nevyhnutné na prenos komunikácie, bezpečnosť alebo poskytnutie funkcie, ktorú ste výslovne požadovali. Takéto technológie môžeme používať bez súhlasu iba v rozsahu, v akom to umožňujú právne predpisy.",
                    "Analytické, preferenčné a marketingové cookies alebo podobné technológie, ktoré nie sú nevyhnutné, používame iba po získaní platného súhlasu. Súhlas môžete spravovať alebo odvolať v nastaveniach cookies, ak sú na webe dostupné; cookies môžete odstrániť aj v nastaveniach prehliadača.",
                    "Technické a bezpečnostné logy uchovávame pri bežnej prevádzke spravidla najviac {days} dní. Dlhšie uchovanie môže byť potrebné na vyšetrenie bezpečnostného incidentu, predchádzanie zneužitiu, ochranu práv alebo splnenie zákonnej povinnosti.".format(days=technical_logs_days),
                ],
            },
            {
                "title": "10. Newsletter a produktové oznámenia",
                "paragraphs": [
                    "Ak sa prihlásite na odber produktových noviniek, použijeme vašu e-mailovú adresu na zasielanie oznámení o aktualizáciách, funkciách a informáciách týkajúcich sa Spectline. Právnym základom je váš súhlas, ak právne predpisy neumožňujú konkrétnu správu zaslať na inom základe.",
                    "Súhlas môžete kedykoľvek odvolať prostredníctvom odkazu na odhlásenie v e-maile alebo kontaktovaním {email}. Odvolanie nemá vplyv na zákonnosť spracúvania pred jeho odvolaním. Po odhlásení môžeme uchovať minimálny záznam potrebný na evidenciu odvolania a zabránenie ďalšiemu zasielaniu.".format(email=email),
                ],
            },
            {
                "title": "11. Príjemcovia a sprostredkovatelia",
                "paragraphs": [
                    "Osobné údaje nepredávame. V nevyhnutnom rozsahu ich môžeme sprístupniť príjemcom alebo sprostredkovateľom, ktorí nám pomáhajú prevádzkovať web a softvér, zabezpečiť komunikáciu, platby a podporu. Dodávateľom umožňujeme prístup iba v rozsahu potrebnom na ich úlohu a na základe príslušných zmluvných a bezpečnostných opatrení.",
                ],
                "bullets": [
                    "Poskytovatelia hostingu, cloudovej infraštruktúry, serverov, databáz, zálohovania a technickej podpory.",
                    "Poskytovatelia autentifikácie, správy používateľských účtov, e-mailovej komunikácie a zákazníckej podpory.",
                    "Poskytovatelia výpočtových, analytických alebo AI technológií, ak sa používajú na poskytovanie funkcií softvéru.",
                    "Banky, poskytovatelia platobných a fakturačných služieb, účtovníci a daňoví poradcovia.",
                    "Poskytovatelia newslettera, analytiky a správy súhlasov s cookies, ak sú tieto služby nasadené.",
                    "MetaTrader, broker alebo iný externý poskytovateľ iba v rozsahu, v akom si aktivujete integráciu alebo je prenos nevyhnutný na vami požadovanú funkciu; ich vlastné spracúvanie sa riadi ich podmienkami.",
                    "Právni a odborní poradcovia, súdy, orgány verejnej moci alebo iné osoby, ak to vyžaduje zákon alebo je to nevyhnutné na ochranu práv.",
                ],
            },
            {
                "title": "12. Prenos osobných údajov mimo EÚ/EHP",
                "paragraphs": [
                    "Ak niektorý z našich dodávateľov alebo technických systémov spracúva osobné údaje mimo Európskeho hospodárskeho priestoru, uskutočníme prenos iba pri splnení podmienok GDPR. Môže ísť najmä o prenos do krajiny, pre ktorú Európska komisia prijala rozhodnutie o primeranej ochrane, alebo o použitie vhodných záruk, napríklad štandardných zmluvných doložiek, spolu s doplňujúcimi opatreniami, ak sú potrebné.",
                    "Informácie o konkrétnych príjemcoch a použitých zárukách vám poskytneme na požiadanie, pokiaľ ich zverejnenie neobmedzuje zákon alebo oprávnené bezpečnostné dôvody.",
                ],
            },
            {
                "title": "13. Doba uchovávania údajov",
                "paragraphs": [
                    "Osobné údaje uchovávame iba po dobu potrebnú na účel, na ktorý boli získané, alebo po dobu vyžadovanú právnymi predpismi. Údaje používateľského účtu a obchodné údaje potrebné na fungovanie služby uchovávame počas aktívneho používania účtu a následne iba po dobu potrebnú na uzatvorenie účtu, vysporiadanie záväzkov, riešenie sporov, ochranu právnych nárokov alebo splnenie zákonných povinností.",
                    "Účtovné a daňové doklady uchovávame počas lehôt stanovených právnymi predpismi. Dopyty bez následnej zmluvy uchovávame spravidla najviac {months} mesiacov od posledného kontaktu a bežné technické logy spravidla najviac {days} dní, ak nie je odôvodnené dlhšie uchovanie. Záložné kópie sa môžu odstrániť v rámci pravidelného cyklu zálohovania.".format(months=inquiry_months, days=technical_logs_days),
                    "Po uplynutí príslušnej doby údaje vymažeme, anonymizujeme alebo bezpečne zneprístupníme, ak ich ďalšie uchovávanie nie je potrebné alebo zákonom vyžadované.",
                ],
            },
            {
                "title": "14. Práva dotknutých osôb",
                "paragraphs": [
                    "Za podmienok stanovených GDPR máte právo požiadať o prístup k osobným údajom, opravu nesprávnych alebo doplnenie neúplných údajov, výmaz, obmedzenie spracúvania a prenosnosť údajov. Ak spracúvanie vykonávame na základe oprávneného záujmu, máte právo namietať. Proti spracúvaniu na účely priameho marketingu môžete namietať kedykoľvek. Ak je právnym základom súhlas, môžete ho kedykoľvek odvolať; odvolanie nemá spätný vplyv na zákonnosť predchádzajúceho spracúvania.",
                    "Žiadosť môžete poslať na {email}. Pred jej vybavením môžeme primerane overiť vašu totožnosť. O žiadosti rozhodneme a odpovieme v lehotách podľa GDPR. Uplatnenie práv je bezplatné, okrem prípadov, keď GDPR umožňuje účtovať primeraný poplatok alebo odmietnuť zjavne neopodstatnenú či neprimeranú žiadosť.".format(email=email),
                ],
            },
            {
                "title": "15. Profilovanie a automatizované spracúvanie",
                "paragraphs": [
                    "AI funkcie Spectline automatizovanými prostriedkami analyzujú údaje o obchodnej histórii a nastaveniach s cieľom identifikovať obchodné vzorce, vyhodnocovať stratégie a vytvárať výstupy pre používateľa. Takéto vyhodnocovanie môže predstavovať profilovanie v zmysle GDPR, ak sa ním automatizovane hodnotia osobné aspekty konkrétnej osoby. Na všeobecnej úrovni systém vyhľadáva štatistické a opakujúce sa vzťahy v údajoch, ktoré mu sprístupníte; výsledok závisí od kvality a rozsahu vstupných údajov a od nastavení softvéru.",
                    "Ak aktivujete automatizované obchodovanie, výstupy softvéru môžu podľa vašej konfigurácie viesť k vytvoreniu alebo odovzdaniu obchodných pokynov. Takéto pokyny môžu ovplyvniť výsledok obchodovania a spôsobiť finančnú stratu. Nejde o rozhodovanie Spectline o vašom právnom postavení alebo o schválení registrácie; registrácia používateľa sa podľa aktuálneho procesu posudzuje administrátorsky.",
                    "Článok 22 GDPR sa uplatní iba vtedy, ak sú splnené jeho zákonné podmienky, najmä ak ide o rozhodnutie založené výlučne na automatizovanom spracúvaní, ktoré má právne účinky alebo vás podobne významne ovplyvňuje. Ak sa takáto operácia v konkrétnej funkcii uplatní, máte práva a záruky vyplývajúce z GDPR, vrátane možnosti požiadať o ľudský zásah, vyjadriť svoje stanovisko a napadnúť rozhodnutie, ak sa na daný prípad vzťahujú.",
                ],
            },
            {
                "title": "16. Bezpečnosť osobných údajov",
                "paragraphs": [
                    "Používame primerané technické a organizačné opatrenia na ochranu osobných údajov pred neoprávneným prístupom, stratou, zmenou, zverejnením alebo iným nezákonným spracúvaním. Podľa povahy systému môžu zahŕňať riadenie prístupov, autentifikáciu, zabezpečený prenos, aktualizácie, zálohovanie, monitorovanie a obmedzenie prístupu na osoby, ktoré údaje potrebujú na plnenie svojich úloh.",
                    "Žiadny spôsob prenosu alebo uchovávania údajov nie je úplne bez rizika. Bezpečnostné opatrenia preto primerane prehodnocujeme s ohľadom na povahu, rozsah, kontext a účely spracúvania a na možné riziká pre vaše práva a slobody.",
                ],
            },
            {
                "title": "17. Sťažnosť dozornému orgánu",
                "paragraphs": [
                    "Ak sa domnievate, že spracúvanie vašich osobných údajov porušuje GDPR alebo slovenské právne predpisy, máte právo podať sťažnosť alebo návrh na začatie konania na Úrad na ochranu osobných údajov Slovenskej republiky. Informácie a kontaktné údaje úradu sú dostupné na dataprotection.gov.sk. Svoje práva môžete uplatniť aj voči príslušnému dozornému orgánu v členskom štáte vášho obvyklého pobytu, pracoviska alebo miesta údajného porušenia, ak to GDPR umožňuje.",
                ],
            },
            {
                "title": "18. Zmeny tohto oznámenia",
                "paragraphs": [
                    "Toto oznámenie môžeme aktualizovať najmä pri zmene funkcií Spectline, spôsobov spracúvania, používaných dodávateľov alebo právnych požiadaviek. Aktuálna verzia bude zverejnená na webe {website}. Ak bude zmena podstatná, upozorníme vás spôsobom primeraným jej významu, ak to bude potrebné.".format(website=website),
                    "Posledná aktualizácia: {updated}.".format(updated=current_date("sk")),
                ],
            },
        ]

    return [
        {
            "title": "1. Controller and contact details",
            "paragraphs": [
                "The entity identified below is the controller of your personal data and determines the purposes and means of processing. If you have questions about this notice or wish to exercise your data protection rights, contact us using the email address listed below.",
            ],
            "bullets": controller_lines("en"),
        },
        {
            "title": "2. Scope of this notice and data sources",
            "paragraphs": [
                "This notice covers personal data processing when you visit our website, create or manage a user account, use Spectline software, use its analytical or automation features, contact us or subscribe to product updates.",
                "We obtain data mainly directly from you when you register, update your account, enter data into the software, contact support or subscribe to updates. Technical and operational records are also generated when you use the website and software. Data from a connected trading platform or broker environment are obtained only to the extent that the particular feature actually makes them available and you have activated or authorised that connection.",
            ],
        },
        {
            "title": "3. Categories of personal data",
            "paragraphs": [
                "The data processed depend on whether you only visit the website, create an account, use trading and AI features or contact us. We process only data appropriate for the relevant purpose.",
            ],
            "bullets": [
                "Identification and contact data: first and last name, email address, address and city; depending on what you provide or what is needed for the account, telephone number, company name and website.",
                "Account and service-use data: account identifier, registration and approval status, account settings, authentication and security records, login history and support requests.",
                "Trading and market data that you upload, enter or make available through an activated integration: trading and order history, traded instruments, timeframes, entry and exit prices, volumes, Stop Loss, Take Profit, timestamps, trading results and trading account or platform identifiers.",
                "Strategy and AI-feature data: trading setups, strategy rules and parameters, bot settings, data used for training or backtesting, analytical outputs, identified patterns, statistics, signals and records of software feature execution.",
                "Technical and operational data: IP address, device type, operating system, browser, session identifiers, access time, error reports, security and system logs and website usage data.",
                "Contract, payment and accounting data: orders, billing details, payment history, subscription information and payment-related communications. Payment card details are generally processed by the relevant payment service provider, where one is used.",
                "Communication and marketing data: messages submitted through the contact form or email, the email address used for product updates and records of granting, changing or withdrawing consent.",
            ],
        },
        {
            "title": "4. Purposes and legal bases",
            "paragraphs": [
                "We process personal data in accordance with Article 6 GDPR and only for specified purposes. Depending on the circumstances, the purposes and legal bases include:",
            ],
            "bullets": [
                "Creating and managing an account, verifying users, handling registration and providing access to service features - performance of a contract or steps taken before entering into a contract under Article 6(1)(b) GDPR.",
                "Providing AI analysis of trading history, identifying patterns, training a strategy for your use, backtesting and providing other software features - performance of a contract under Article 6(1)(b) GDPR.",
                "Processing orders, subscriptions, payments, invoices and accounting and complying with tax or other statutory duties - performance of a contract and compliance with a legal obligation under Article 6(1)(b) and (c) GDPR.",
                "Operating, maintaining and troubleshooting the service, securing accounts and systems, preventing fraud and misuse and protecting legal claims - legitimate interests under Article 6(1)(f) GDPR, subject to the required assessment of our interests and your rights.",
                "Responding to general questions and handling enquiries before a contract is concluded - pre-contractual steps or, depending on the circumstances, our legitimate interest in proper business communications.",
                "Sending voluntary product updates and using non-essential analytics or marketing cookies - your consent under Article 6(1)(a) GDPR, which you may withdraw at any time.",
            ],
        },
        {
            "title": "5. User accounts and software services",
            "paragraphs": [
                "When you register for and manage an account, we process the data needed to create the account, communicate with you, manage access rights, carry out any administrator approval of registration, provide customer support and perform our contract. In the current registration process, the required information includes your first name, last name, email address, address, city and authentication details; without mandatory information, we may be unable to create the account or provide the service. Telephone number, company name and website are provided only where the form requires them or you choose to add them. You may choose not to provide fields identified as optional.",
                "We also use account data to store your settings and maintain continuity when you use the software. If you close your account, we will stop or restrict processing according to the relevant purpose, legal obligations, retention rules and the need to protect legal claims.",
            ],
        },
        {
            "title": "6. Trading history, AI analysis and automated trading",
            "paragraphs": [
                "If you use Spectline trading features, we process your trading history and settings to analyse trading patterns, train or configure a strategy for your use, perform backtesting, evaluate results and provide software outputs. The specific scope depends on the features you activate and the data you enter or make available to the service.",
                "If you activate automated trading and authorise a connection to a compatible platform, the software may, according to your settings, create or transmit trading instructions to the relevant environment. This processing may have financial consequences. Trading history and AI outputs are used to provide the service features; using them for an unrelated or incompatible purpose, or to train a general model outside your account, would require a separately assessed legal basis and appropriate information.",
                "MetaTrader 4, MetaTrader 5 and the broker where you hold a trading account may also process data for their own purposes and under their own privacy notices. Whether and which data are exchanged between Spectline, a platform and a broker depends on the specific technical integration. We do not automatically disclose data to these entities merely because their products are mentioned on our website.",
            ],
        },
        {
            "title": "7. Third-party data and safe use",
            "paragraphs": [
                "Only enter data into the software that you are entitled to use and disclose. If trading or other uploaded files contain personal data relating to other individuals, you are responsible for having a legal basis to provide them and for meeting the relevant transparency obligations. If Spectline processes personal data of other individuals solely on your instructions as a processor, the terms of that processing will, where required, be set out in a separate agreement under Article 28 GDPR.",
                "Do not send broker account passwords, private keys or other secret authentication information through the contact form or ordinary email. If a particular integration requires an authorisation token or another access credential, its processing must match the technical integration and the security information provided when the integration is activated.",
                "The service is not designed to process special categories of personal data under Article 9 GDPR. Do not upload such data unless a separate legal basis and an agreed procedure are in place.",
            ],
        },
        {
            "title": "8. Contact form and enquiries",
            "paragraphs": [
                "When you contact us through the form or by email, we process the information you provide, in particular your name, email address, subject and message content. We use it to respond, handle your request and, where relevant, take steps before entering into a contract.",
                "If an enquiry does not result in a contractual relationship, we normally retain the correspondence for no longer than {months} months after the last contact, unless longer retention is necessary to protect legal claims or for another lawful reason.".format(months=inquiry_months),
            ],
        },
        {
            "title": "9. Cookies and technical logs",
            "paragraphs": [
                "The website may use technical cookies or similar technologies that are necessary to transmit a communication, ensure security or provide a feature you have expressly requested. We may use such technologies without consent only to the extent permitted by applicable law.",
                "Non-essential analytics, preference and marketing cookies or similar technologies are used only after valid consent has been obtained. You may manage or withdraw consent in the cookie settings, where available, and you can also remove cookies through your browser settings.",
                "During normal operation, technical and security logs are generally retained for no longer than {days} days. Longer retention may be necessary to investigate a security incident, prevent misuse, protect rights or comply with a legal obligation.".format(days=technical_logs_days),
            ],
        },
        {
            "title": "10. Newsletter and product updates",
            "paragraphs": [
                "If you subscribe to product updates, we use your email address to send information about Spectline updates, features and related announcements. The legal basis is your consent, unless applicable law permits a particular communication on another basis.",
                "You may withdraw consent at any time by using the unsubscribe link in an email or by contacting {email}. Withdrawal does not affect the lawfulness of processing before withdrawal. After unsubscribing, we may retain a minimal record of the withdrawal to demonstrate compliance and prevent further messages.".format(email=email),
            ],
        },
        {
            "title": "11. Recipients and processors",
            "paragraphs": [
                "We do not sell personal data. Where necessary, we may disclose data to recipients or processors who help us operate the website and software, provide communications, payments and support. Suppliers are given access only to the extent needed for their task and subject to appropriate contractual and security measures.",
            ],
            "bullets": [
                "Hosting, cloud infrastructure, server, database, backup and technical support providers.",
                "Authentication, user account management, email communication and customer support providers.",
                "Computing, analytics or AI technology providers, where used to provide software features.",
                "Banks, payment and invoicing providers, accountants and tax advisers.",
                "Newsletter, analytics and cookie-consent management providers, where these services are deployed.",
                "MetaTrader, a broker or another external provider only to the extent that you activate an integration or the transfer is necessary for a feature you request; their own processing is governed by their terms.",
                "Legal and professional advisers, courts, public authorities or other persons where required by law or necessary to protect rights.",
            ],
        },
        {
            "title": "12. Transfers outside the EU/EEA",
            "paragraphs": [
                "If one of our suppliers or technical systems processes personal data outside the European Economic Area, we will make the transfer only where the GDPR requirements are met. This may include a transfer to a country covered by a European Commission adequacy decision or the use of appropriate safeguards, such as Standard Contractual Clauses, together with supplementary measures where required.",
                "Information about specific recipients and safeguards may be requested from us, subject to legal restrictions and legitimate security considerations.",
            ],
        },
        {
            "title": "13. Data retention",
            "paragraphs": [
                "We retain personal data only for as long as necessary for the purpose for which they were collected or for as long as required by law. User account data and trading data needed to operate the service are retained while the account is active and afterwards only for the period needed to close the account, settle outstanding obligations, resolve disputes, protect legal claims or comply with legal duties.",
                "Accounting and tax records are retained for the periods required by law. Enquiries that do not lead to a contract are normally retained for no longer than {months} months after the last contact, and ordinary technical logs for no longer than {days} days, unless longer retention is justified. Backup copies may be removed as part of the regular backup cycle.".format(months=inquiry_months, days=technical_logs_days),
                "When the applicable retention period expires, we delete, anonymise or securely restrict access to the data unless further retention is necessary or required by law.",
            ],
        },
        {
            "title": "14. Data subject rights",
            "paragraphs": [
                "Subject to the conditions set out in the GDPR, you may request access to your personal data, rectification of inaccurate data or completion of incomplete data, erasure, restriction of processing and data portability. Where processing is based on our legitimate interests, you may object. You may object at any time to processing for direct marketing. Where processing is based on consent, you may withdraw it at any time; withdrawal does not affect the lawfulness of processing carried out before it.",
                "Requests may be sent to {email}. We may reasonably verify your identity before handling a request. We will respond within the time limits set by the GDPR. Exercising your rights is free of charge, except where the GDPR permits a reasonable fee or refusal of a manifestly unfounded or excessive request.".format(email=email),
            ],
        },
        {
            "title": "15. Profiling and automated processing",
            "paragraphs": [
                "Spectline AI features use automated means to analyse trading history and settings in order to identify trading patterns, evaluate strategies and generate outputs for the user. This evaluation may constitute profiling under the GDPR where automated processing is used to evaluate personal aspects of an identifiable individual. At a high level, the system searches for statistical and recurring relationships in the data you make available; results depend on the quality and volume of input data and the software settings.",
                "If you activate automated trading, software outputs may, according to your configuration, result in the creation or transmission of trading instructions. Such instructions may affect trading results and cause financial loss. Spectline does not use this process to determine your legal status or approve your registration; under the current process, user registration is reviewed by an administrator.",
                "Article 22 GDPR applies only where its legal conditions are met, in particular where a decision based solely on automated processing produces legal effects or similarly significantly affects you. If a particular feature involves such processing, you have the rights and safeguards provided by the GDPR, including, where applicable, the right to obtain human intervention, express your point of view and contest the decision.",
            ],
        },
        {
            "title": "16. Data security",
            "paragraphs": [
                "We use appropriate technical and organisational measures to protect personal data against unauthorised access, loss, alteration, disclosure or other unlawful processing. Depending on the system, measures may include access management, authentication, secure transmission, updates, backups, monitoring and limiting access to persons who need the data for their duties.",
                "No method of transmitting or storing data is completely risk-free. We therefore review security measures as appropriate to the nature, scope, context and purposes of processing and the risks to your rights and freedoms.",
            ],
        },
        {
            "title": "17. Complaint to a supervisory authority",
            "paragraphs": [
                "If you believe that the processing of your personal data infringes the GDPR or Slovak data protection law, you have the right to lodge a complaint or initiate proceedings with the Office for Personal Data Protection of the Slovak Republic. Information and contact details are available at dataprotection.gov.sk. Where permitted by the GDPR, you may also contact the supervisory authority in the EU Member State of your habitual residence, place of work or the place of the alleged infringement.",
            ],
        },
        {
            "title": "18. Changes to this notice",
            "paragraphs": [
                "We may update this notice, particularly when Spectline features, processing methods, suppliers or legal requirements change. The current version will be published at {website}. If a change is material, we will notify you in a manner appropriate to its significance where required.".format(website=website),
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