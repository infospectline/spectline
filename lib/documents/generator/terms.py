# lib/documents/generator/terms.py
"""Generate Slovak and English Spectline software terms as PDFs.

IMPORTANT BEFORE PUBLICATION:
- Complete the Provider's legal identity in DATA['provider'] (full legal name,
  full postal address, company/entrepreneur ID and any applicable tax details).
- Confirm that the actual Spectline product workflow is accurately described.
- Obtain Slovak/EU financial-regulatory review of the automated-trading features.
- Align the checkout flow, subscription/payment flow and privacy notice with these Terms.

This source is a drafting aid, not a legal opinion or confirmation that the product
may be offered without an authorization. The legal classification depends on the
actual functions and the way they are provided, not only on the wording here.
"""

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
# <project>/lib/documents/generator/terms.py
# <project>/public/images/logo_black.png
# <project>/public/fonts/Roboto-Regular.ttf
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


# -----------------------------------------------------------------------------
# PROVIDER DATA
# Do not publish until all legally required trader/provider identification data
# is completed and verified. "Spectline" may be a brand, not the legal person.
# -----------------------------------------------------------------------------
DATA = {
    "version": "3.0",
    "website": "spectline.com",
    "provider": {
        "brand_name": "Spectline",
        # Replace with the exact registered business name / full name of the
        # entrepreneur. Do not leave the brand name here unless it is the legal name.
        "legal_name": "Spectline",
        # Replace with the complete address, including street, number, postcode
        # and municipality, as applicable to the legal form of the Provider.
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
    "commercial_defaults": {
        "payment_due_days": 14,
    },
}


LANGUAGES = {
    "sk": {
        "title": "PODMIENKY POUŽÍVANIA A OBCHODNÉ PODMIENKY SPECTLINE",
        "intro": (
            "Tieto podmienky upravujú prístup k softvéru Spectline a jeho používanie, "
            "vrátane funkcií umelej inteligencie na analýzu historických obchodných dát, "
            "prípravu a testovanie obchodných stratégií a integráciu s platformami "
            "MetaTrader 4 a MetaTrader 5. Ak používateľ aktivuje automatizované "
            "obchodovanie, softvér môže podľa nastavení používateľa vytvárať a odosielať "
            "obchodné pokyny prostredníctvom jeho účtu u brokera. Obchodovanie je rizikové "
            "a môže viesť k strate peňazí; zisk ani konkrétny výsledok nie je zaručený."
        ),
        "last_updated": "Posledná aktualizácia",
        "version": "Verzia",
        "website": "Web",
        "page": "Strana",
        "provider_labels": {
            "legal_name": "Obchodné meno / meno a priezvisko",
            "brand_name": "Obchodné označenie",
            "address": "Sídlo / miesto podnikania",
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
        "title": "SPECTLINE SOFTWARE TERMS AND CONDITIONS",
        "intro": (
            "These Terms govern access to and use of Spectline software, including its "
            "artificial-intelligence features for analysing historical trading data, "
            "preparing and testing trading strategies, and integrating with MetaTrader 4 "
            "and MetaTrader 5. If the user enables automated trading, the software may "
            "generate and submit trading instructions through the user's broker account "
            "according to the user's settings. Trading is risky and may result in loss; "
            "no profit or particular outcome is guaranteed."
        ),
        "last_updated": "Last updated",
        "version": "Version",
        "website": "Website",
        "page": "Page",
        "provider_labels": {
            "legal_name": "Legal name / full name",
            "brand_name": "Trading name",
            "address": "Registered office / business address",
            "country": "Country",
            "ico": "Company / business ID",
            "dic": "Tax ID",
            "ic_dph": "VAT ID",
            "registration": "Registration",
            "email": "Email",
            "phone": "Phone",
            "website": "Website",
        },
    },
}


SECTIONS = {
    "sk": [
        {
            "title": "1. Rozsah pôsobnosti a zmluvné dokumenty",
            "paragraphs": [
                "Tieto podmienky sa vzťahujú na používanie softvéru Spectline, súvisiacich aplikácií, používateľského účtu, dokumentácie, aktualizácií a podporovaných integračných funkcií (ďalej spolu len „Služba“).",
                "Zmluva medzi Poskytovateľom a Používateľom sa skladá z týchto podmienok, informácií zobrazených pri objednávke alebo aktivácii, potvrdenia objednávky, zvoleného balíka, prípadnej samostatnej zmluvy a dokumentácie produktu. Individuálne dojednanie má prednosť pred týmito podmienkami v rozsahu, v akom je s nimi v rozpore.",
                "Pred dokončením objednávky má Používateľ možnosť oboznámiť sa s cenou, rozsahom funkcií, dobou prístupu, prípadným opakovaným účtovaním a ďalšími podstatnými parametrami konkrétnej ponuky. Rozhodujúce sú údaje potvrdené pri danej objednávke.",
            ],
        },
        {
            "title": "2. Poskytovateľ a kontaktné údaje",
            "paragraphs": [
                "Poskytovateľom Služby a zmluvnou stranou je osoba identifikovaná v nasledujúcich údajoch. Obchodné označenie Spectline je značkou Služby; samo osebe nenahrádza zákonné identifikačné údaje podnikateľa alebo právnickej osoby.",
            ],
            "bullets": ["{{provider_lines_sk}}"],
        },
        {
            "title": "3. Definície",
            "paragraphs": ["Na účely týchto podmienok:"],
            "bullets": [
                "„Poskytovateľ“ je osoba uvedená v článku 2.",
                "„Používateľ“ je fyzická alebo právnická osoba, ktorá si vytvorí účet, objedná Službu alebo ju používa.",
                "„Softvér“ alebo „Spectline“ je softvérové riešenie vrátane jeho AI funkcií, analytických, tréningových, testovacích a integračných komponentov sprístupnených v rámci zvoleného balíka.",
                "„Obchodné dáta“ sú historické alebo aktuálne údaje o obchodoch, účtoch, cenách, nastaveniach a ďalšie údaje, ktoré Používateľ poskytne alebo sprístupní Softvéru.",
                "„Broker“ je tretia strana, ktorá vedie obchodný účet Používateľa alebo poskytuje prístup na trh.",
                "„Obchodná platforma“ je najmä MetaTrader 4 alebo MetaTrader 5 a jej príslušné technické rozhrania.",
                "„Automatizované obchodovanie“ je funkcia, pri ktorej Softvér po aktivácii Používateľom môže podľa dostupných nastavení a pravidiel vytvárať alebo odosielať obchodné pokyny prostredníctvom pripojeného účtu.",
                "„Spotrebiteľ“ je fyzická osoba, ktorá pri uzatváraní a plnení zmluvy nekoná v rámci svojej podnikateľskej činnosti alebo povolania.",
            ],
        },
        {
            "title": "4. Predmet Služby a jej funkcie",
            "paragraphs": [
                "Služba poskytuje prístup k softvérovým funkciám Spectline v rozsahu konkrétneho zakúpeného alebo aktivovaného balíka. Podľa dostupnej verzie môže zahŕňať import alebo spracovanie Obchodných dát, analýzu obchodných vzorcov pomocou AI, tréning alebo testovanie stratégie, zobrazovanie analytických výstupov a integráciu s MetaTrader 4 alebo MetaTrader 5.",
                "Ak Používateľ aktivuje funkciu automatizovaného obchodovania a dokončí potrebné prepojenie, Softvér môže generovať a odosielať obchodné pokyny na účet Používateľa podľa zvolených alebo potvrdených nastavení. Používateľ berie na vedomie, že takéto pokyny môžu byť odoslané bez samostatného manuálneho potvrdenia každého jednotlivého obchodu.",
                "Presný rozsah funkcií, podporované platformy, technické požiadavky, dostupné trhy a prípadné limity určuje aktuálna dokumentácia a ponuka konkrétneho balíka. Funkcia, ktorá nie je v balíku alebo dokumentácii výslovne uvedená, sa nepovažuje za súčasť Služby.",
            ],
        },
        {
            "title": "5. Regulačný charakter činnosti a hranice Služby",
            "paragraphs": [
                "Tieto podmienky upravujú sprístupnenie a používanie Softvéru. Samotné označenie produktu ako softvéru alebo technického nástroja však nerozhoduje o právnej klasifikácii jeho skutočných funkcií. Rozhodujúci je reálny spôsob fungovania, miera rozhodovacej voľnosti, oprávnenia k účtu Používateľa a spôsob, akým sa pokyny vytvárajú a vykonávajú.",
                "Poskytovateľ neposkytuje na základe týchto podmienok osobné investičné odporúčania zohľadňujúce individuálnu finančnú situáciu Používateľa, ani sa týmito podmienkami neuzatvára zmluva o riadení portfólia alebo o vykonávaní pokynov klienta. Žiadne ustanovenie však nemení právnu povahu činnosti, ak by konkrétna funkcia alebo spôsob jej prevádzky podľa záväzných právnych predpisov predstavovali regulovanú investičnú službu alebo činnosť.",
                "Funkcia, ktorá by predstavovala regulovanú investičnú službu alebo činnosť, môže byť poskytovaná len vtedy a takým subjektom, ktorý na to má potrebné oprávnenie alebo sa môže oprieť o uplatniteľnú zákonnú výnimku. Ak je to potrebné, jej poskytovanie sa upraví samostatnou zmluvou a príslušnými regulačnými dokumentmi. Tieto podmienky samy osebe nie sú povolením na poskytovanie investičných služieb.",
                "Používateľ nesmie považovať analytické výstupy, signály, skóre, spätné testy ani správanie AI modelu za osobné investičné poradenstvo alebo garanciu vhodnosti konkrétneho obchodu pre jeho situáciu.",
            ],
        },
        {
            "title": "6. Upozornenie na riziká obchodovania",
            "paragraphs": [
                "Obchodovanie s finančnými nástrojmi, menami, CFD, derivátmi alebo inými produktmi dostupnými cez Brokera je rizikové. Riziko je osobitne vysoké pri produktoch s finančnou pákou. Hodnota účtu môže rýchlo klesnúť a Používateľ môže stratiť časť alebo celý kapitál určený na obchodovanie. V závislosti od produktu, zmluvy s Brokerom a uplatniteľnej ochrany môže strata za určitých okolností presiahnuť pôvodne vyčlenenú sumu.",
                "Používateľ obchoduje na vlastné riziko a mal by obchodovať iba s prostriedkami, ktorých stratu si môže dovoliť. Pred použitím reálneho účtu by mal Softvér a nastavenia dôkladne otestovať na demo účte, porozumieť obchodovaným produktom a nezávisle posúdiť riziko stratégie.",
                "Žiadna stratégia, algoritmus, AI model, historický výsledok ani spätný test neodstraňuje trhové riziko. Stop Loss, Take Profit ani iné ochranné nastavenie nemusia byť vykonané za očakávanú cenu alebo v očakávanom čase, napríklad pri cenových medzerách, nízkej likvidite, volatilite alebo výpadku spojenia.",
                "Používateľ by nemal aktivovať automatizované obchodovanie, ak nerozumie jeho fungovaniu, rizikovým parametrom a spôsobu zastavenia obchodovania.",
            ],
        },
        {
            "title": "7. AI analýza, historické dáta a výkonnosť",
            "paragraphs": [
                "AI funkcie analyzujú Obchodné dáta a môžu identifikovať štatistické vzorce alebo vytvárať výstupy podľa dostupných dát, modelov a nastavení. Výsledok závisí od kvality, úplnosti, reprezentatívnosti a správnosti vstupných dát, ako aj od použitej metodiky a podmienok trhu.",
                "Historická výkonnosť, spätné testovanie, simulácie, tréningové výsledky ani výsledky iných používateľov nie sú spoľahlivým ukazovateľom budúcej výkonnosti. Výsledky simulácie sa môžu významne líšiť od reálneho obchodovania v dôsledku spreadu, komisií, swapov, sklzu, latencie, likvidity, dostupnosti cien, výpadkov alebo zmien trhových podmienok.",
                "Poskytovateľ nezaručuje zisk, konkrétnu mieru úspešnosti, presnosť predikcií, počet obchodov, minimálny výnos, pasívny príjem ani to, že Softvér zabráni strate. Akýkoľvek príklad, graf alebo historický výsledok má iba informačný alebo ilustračný charakter, pokiaľ nie je výslovne a zákonne dohodnuté inak.",
                "AI modely a softvérové výstupy môžu obsahovať chyby, skreslenie, neúplnosť alebo nepresnosti. Poskytovateľ môže meniť modely a algoritmy pri zachovaní dohodnutých vlastností Služby a príslušných práv Používateľa.",
            ],
        },
        {
            "title": "8. Používateľský účet a oprávnenosť používania",
            "paragraphs": [
                "Používateľ je povinný pri registrácii a používaní Služby uvádzať pravdivé, úplné a aktuálne údaje a udržiavať ich aktuálne. Používať Službu môže iba osoba, ktorá má právnu spôsobilosť uzatvoriť zmluvu a je oprávnená používať príslušné obchodné a brokerské služby vo svojej jurisdikcii.",
                "Používateľ zodpovedá za ochranu prihlasovacích údajov, hesiel, prístupových tokenov a ďalších autentifikačných prostriedkov. Bezodkladne oznámi Poskytovateľovi podozrenie na neoprávnený prístup alebo bezpečnostný incident.",
                "Používateľ nesmie sprístupniť účet inej osobe, obchádzať bezpečnostné alebo licenčné obmedzenia ani umožniť používanie Služby spôsobom, ktorý porušuje tieto podmienky alebo práva tretích osôb.",
            ],
        },
        {
            "title": "9. Brokeri a integrácia s MetaTrader 4 / MetaTrader 5",
            "paragraphs": [
                "Na využitie niektorých funkcií môže byť potrebný samostatný obchodný účet u Brokera, obchodná platforma MetaTrader 4 alebo MetaTrader 5 a vhodné internetové pripojenie. Účet u Brokera, jeho financovanie, obchodné podmienky, poplatky, maržové požiadavky, realizácia pokynov a ochrana klienta sa riadia samostatnou zmluvou medzi Používateľom a Brokerom.",
                "Poskytovateľ nie je zmluvnou stranou vzťahu medzi Používateľom a Brokerom a nezodpovedá za rozhodnutia Brokera, jeho obchodné podmienky, dostupnosť, odmietnutie alebo oneskorenie pokynu, cenu realizácie, výpadky ani zmeny jeho platformy, s výnimkou zodpovednosti, ktorú nemožno podľa právnych predpisov vylúčiť.",
                "MetaTrader, MetaQuotes a názvy Brokerov sú označeniami ich príslušných vlastníkov. Ich uvedenie slúži na identifikáciu kompatibilnej platformy a samo osebe neznamená vlastnícke prepojenie, schválenie ani partnerstvo s Poskytovateľom.",
                "Kompatibilita sa môže líšiť podľa verzie platformy, Brokera, typu účtu, operačného systému, servera a technických rozhraní. Poskytovateľ nezaručuje podporu každej konfigurácie alebo budúcej verzie tretej strany.",
            ],
        },
        {
            "title": "10. Povinnosti a zodpovednosť Používateľa pri obchodovaní",
            "paragraphs": [
                "Používateľ je zodpovedný za výber Brokera a obchodovaného produktu, kontrolu oprávnenosti obchodovania, nastavenie účtu, kapitálu, finančnej páky, objemu pozícií, rizikových limitov a za rozhodnutie, či aktivuje automatizované obchodovanie.",
                "Pred aktiváciou automatizovaného obchodovania Používateľ skontroluje parametre stratégie, podporované symboly a časové rámce, veľkosť pozície, Stop Loss, Take Profit, maximálnu prípustnú stratu a ďalšie nastavenia, ktoré sú v produkte dostupné. Používateľ je povinný priebežne primerane kontrolovať činnosť Softvéru a vedieť automatizované obchodovanie pozastaviť alebo vypnúť.",
                "Používateľ nesmie používať Službu na manipuláciu trhu, podvod, pranie špinavých peňazí, obchádzanie sankcií, porušovanie pravidiel Brokera, obchodných pravidiel alebo právnych predpisov. Používateľ nesie zodpovednosť za svoje daňové a regulačné povinnosti súvisiace s obchodovaním.",
                "Poskytovateľ nie je povinný sledovať obchodný účet Používateľa, upozorňovať na každý obchod alebo zasiahnuť do stratégie, pokiaľ taká povinnosť nie je výslovne dohodnutá v samostatnej zmluve a právne prípustná.",
            ],
        },
        {
            "title": "11. Obchodné dáta, vstupy a oprávnenie na ich spracovanie",
            "paragraphs": [
                "Používateľ zodpovedá za to, že je oprávnený Obchodné dáta a ďalšie materiály do Služby nahrať, sprístupniť a používať na účely dohodnuté v týchto podmienkach. Používateľ nesmie poskytnúť dáta, ktorých spracovanie by porušovalo práva inej osoby, zmluvu s Brokerom alebo právne predpisy.",
                "Používateľ udeľuje Poskytovateľovi nevýhradné, územne neobmedzené oprávnenie v rozsahu potrebnom na prevádzku Služby ukladať, kopírovať, prenášať, analyzovať a technicky upravovať poskytnuté dáta na účely sprístupnenia funkcií, podpory, zabezpečenia a plnenia zmluvy. Toto oprávnenie neprevádza vlastníctvo dát na Poskytovateľa.",
                "Použitie dát na samostatné účely, ktoré presahujú poskytovanie Služby, najmä na tréning všeobecných alebo zdieľaných modelov, marketing alebo sprístupnenie tretím osobám, musí byť Používateľovi transparentne oznámené a musí mať samostatný platný právny základ alebo súhlas, ak ho právne predpisy vyžadujú.",
                "Používateľ nesmie do Služby nahrávať nepotrebné osobné údaje tretích osôb, prihlasovacie údaje umožňujúce neoprávnené nakladanie s finančnými prostriedkami ani škodlivý kód.",
            ],
        },
        {
            "title": "12. Dostupnosť, údržba, aktualizácie a technické obmedzenia",
            "paragraphs": [
                "Poskytovateľ vynaloží primerané úsilie na prevádzku a údržbu Služby. Nezaručuje však nepretržitú dostupnosť, bezchybnosť, okamžitú odozvu ani kompatibilitu so všetkými zariadeniami, sieťami, systémami, dátovými zdrojmi alebo službami tretích strán.",
                "Služba môže byť dočasne obmedzená alebo nedostupná z dôvodu údržby, aktualizácie, bezpečnostného zásahu, výpadku internetu, cloudu, dátového zdroja, Brokera, Obchodnej platformy alebo inej okolnosti mimo primeranej kontroly Poskytovateľa. Ak je to možné, Poskytovateľ oznámi plánovanú údržbu vopred.",
                "Poskytovateľ môže vydávať opravy, aktualizácie a zmeny funkcií. Pri digitálnom plnení poskytovanom Spotrebiteľovi sa aktualizácie a zmeny vykonávajú v súlade s kogentnými požiadavkami právnych predpisov na súlad digitálneho plnenia a informovanie Spotrebiteľa.",
            ],
        },
        {
            "title": "13. Cena, platby a prípadné predplatné",
            "paragraphs": [
                "Používateľ zaplatí cenu uvedenú v objednávke, cenovej ponuke alebo potvrdení objednávky. Cena, mena, zahrnuté funkcie, fakturačné obdobie, dane a prípadné ďalšie poplatky musia byť Používateľovi zobrazené pred odoslaním záväznej objednávky.",
                "Ak faktúra alebo individuálna dohoda neurčuje inak, lehota splatnosti faktúry je {{due_days}} dní odo dňa jej vystavenia. Poskytovateľ vystaví účtovný doklad v súlade s uplatniteľnými právnymi predpismi.",
                "Ak konkrétny balík funguje na báze predplatného alebo opakovaných platieb, cena, dĺžka fakturačného obdobia, dátum a spôsob obnovy, podmienky zmeny ceny a postup zrušenia predplatného budú uvedené pred uzatvorením objednávky. Opakované účtovanie sa uplatní iba vtedy, ak bolo v objednávke jasne uvedené a Používateľ ho odsúhlasil.",
                "Pri omeškaní s úhradou môže Poskytovateľ po predchádzajúcom upozornení primerane obmedziť alebo pozastaviť prístup k platenej Službe, ak to povoľujú právne predpisy a nie sú tým dotknuté kogentné práva Spotrebiteľa. Zákonné úroky z omeškania možno požadovať len v rozsahu dovolenom právnymi predpismi.",
            ],
        },
        {
            "title": "14. Licencia a duševné vlastníctvo",
            "paragraphs": [
                "Ak objednávka neurčuje inak, Poskytovateľ udeľuje Používateľovi počas zaplatenej a dohodnutej doby nevýhradnú, neprenosnú a nesublicencovateľnú licenciu používať Softvér na vlastné interné účely a v rozsahu funkcií príslušného balíka.",
                "Všetky práva k Softvéru, zdrojovému kódu, modelom, algoritmom, rozhraniam, dokumentácii, dizajnu, značke, databázam a súvisiacim materiálom zostávajú Poskytovateľovi alebo príslušným tretím stranám. Používateľ nadobúda iba práva výslovne udelené týmito podmienkami alebo samostatnou licenčnou zmluvou.",
                "Používateľ nesmie Softvér kopírovať, predávať, prenajímať, sublicencovať, sprístupňovať ako službu tretej osobe, obchádzať technické ochrany, spätne analyzovať alebo dekompilovať, okrem rozsahu, v ktorom taký zákaz nepripúšťa obmedziť kogentný právny predpis.",
                "Práva k MetaTraderu, dátovým zdrojom, knižniciam, pluginom a iným komponentom tretích strán sa riadia ich vlastnými licenčnými podmienkami.",
            ],
        },
        {
            "title": "15. Zakázané používanie a bezpečnosť",
            "paragraphs": [
                "Používateľ nesmie používať Službu na nezákonné účely, narúšať jej prevádzku, skúšať neoprávnený prístup, distribuovať škodlivý kód, obchádzať limity alebo bezpečnostné opatrenia, zasahovať do účtov iných používateľov ani preťažovať infraštruktúru.",
                "Poskytovateľ môže prijať primerané bezpečnostné opatrenia, obmedziť prístup alebo pozastaviť účet, ak má dôvodné podozrenie na bezpečnostný incident, podvod, porušenie právnych predpisov alebo týchto podmienok. Ak to okolnosti umožňujú, Používateľa o dôvode informuje.",
            ],
        },
        {
            "title": "16. Tretie strany a externé služby",
            "paragraphs": [
                "Služba môže závisieť od infraštruktúry, cloudových poskytovateľov, dátových zdrojov, platobných služieb, brokerov, obchodných platforiem alebo iných externých služieb. Ich dostupnosť a používanie sa riadia podmienkami príslušnej tretej strany.",
                "Poskytovateľ nezodpovedá za obsah, podmienky, bezpečnosť ani dostupnosť samostatnej služby tretej strany v rozsahu, v akom ich nemá pod kontrolou. Tým nie je dotknutá zodpovednosť Poskytovateľa za vlastné porušenie zmluvných alebo zákonných povinností.",
            ],
        },
        {
            "title": "17. Spotrebiteľské práva, odstúpenie od zmluvy a digitálne plnenie",
            "paragraphs": [
                "Ak je Používateľ Spotrebiteľom a zmluva bola uzavretá na diaľku alebo mimo prevádzkových priestorov Poskytovateľa, Spotrebiteľ má právo odstúpiť od zmluvy v lehote a za podmienok stanovených zákonom, spravidla do 14 dní od uzavretia zmluvy o službe alebo zmluvy o dodávaní digitálneho obsahu. Uplatniteľné výnimky a začiatok plynutia lehoty závisia od právnej povahy konkrétneho plnenia.",
                "Ak má byť dodávanie digitálneho obsahu alebo poskytovanie digitálnej služby začaté pred uplynutím lehoty na odstúpenie, Poskytovateľ si vyžiada výslovnú žiadosť alebo súhlas Spotrebiteľa a poskytne mu zákonom požadované poučenia a potvrdenia. Samotné ustanovenie týchto podmienok nenahrádza samostatný súhlas alebo potvrdenie, ktoré musí byť získané v objednávkovom procese.",
                "Právo na odstúpenie môže zaniknúť alebo byť obmedzené iba v prípadoch a za podmienok výslovne stanovených zákonom, napríklad pri dodaní digitálneho obsahu po splnení zákonných podmienok výslovného súhlasu, poučenia a potvrdenia. Pri službe sa uplatnia odlišné zákonné pravidlá. Používateľovi zostávajú zachované všetky kogentné spotrebiteľské práva.",
                "Oznámenie o odstúpení možno zaslať na {{email}} alebo iným jednoznačným vyhlásením doručeným Poskytovateľovi. Spotrebiteľ môže použiť aj vzorový formulár na odstúpenie, ak mu bol poskytnutý.",
                "Pri digitálnom plnení poskytovanom Spotrebiteľovi má Spotrebiteľ zákonné práva týkajúce sa súladu digitálneho plnenia so zmluvou, jeho funkčnosti, kompatibility, aktualizácií a nápravy. Tieto práva nemožno týmito podmienkami vylúčiť ani obmedziť.",
            ],
        },
        {
            "title": "18. Reklamácie a uplatnenie práv z vadného plnenia",
            "paragraphs": [
                "Používateľ môže oznámiť technický problém, výpadok alebo nesúlad Služby s dohodnutým opisom na adrese {{email}}. Oznámenie by malo obsahovať identifikáciu účtu, opis problému, čas jeho vzniku a dostupné technické informácie. Používateľ by nemal posielať heslá ani celé prístupové údaje k brokerskému účtu.",
                "Poskytovateľ preverí oznámenie a podľa povahy problému navrhne primeraný postup, napríklad opravu, obnovenie prístupu, opätovné poskytnutie digitálneho plnenia alebo iný spôsob nápravy podľa zmluvy a právnych predpisov.",
                "Pri Spotrebiteľovi sa uplatnia zákonné pravidlá zodpovednosti za vady a súladu digitálneho plnenia vrátane lehôt a nápravných prostriedkov. Tieto podmienky neobmedzujú zákonné práva Spotrebiteľa. Pri Používateľovi, ktorý nie je Spotrebiteľom, sa práva z vád posudzujú podľa zmluvy a príslušných právnych predpisov.",
                "Samotná obchodná strata, nevýhodný výsledok stratégie alebo odlišnosť reálneho obchodovania od historického testu nie sú automaticky technickou vadou Softvéru. Tým nie sú dotknuté práva Používateľa, ak problém vznikol porušením povinnosti Poskytovateľa alebo vadou Služby.",
            ],
        },
        {
            "title": "19. Osobné údaje a dôvernosť",
            "paragraphs": [
                "Spracúvanie osobných údajov sa riadi samostatným dokumentom Ochrana osobných údajov dostupným na webe {{website}}. Tento dokument má vysvetľovať najmä účely, právne základy, kategórie údajov, príjemcov, dobu uchovávania a práva dotknutých osôb.",
                "Ak Poskytovateľ spracúva osobné údaje v mene Používateľa ako sprostredkovateľ, strany uzatvoria dohodu o spracúvaní osobných údajov podľa článku 28 GDPR, ak je taká dohoda vzhľadom na skutočný spôsob spracúvania potrebná.",
                "Strany budú chrániť neverejné dôverné informácie, ktoré získajú pri používaní Služby, a použijú ich iba na účely plnenia zmluvy, plnenia zákonných povinností alebo ochrany svojich práv. Táto povinnosť sa nevzťahuje na informácie verejne známe bez porušenia povinnosti alebo na informácie, ktoré musia byť sprístupnené podľa zákona.",
            ],
        },
        {
            "title": "20. Pozastavenie a ukončenie prístupu",
            "paragraphs": [
                "Používateľ môže ukončiť používanie Služby a zrušiť predplatné spôsobom uvedeným v objednávke alebo používateľskom účte. Ak ide o predplatné, zrušenie zabráni ďalšiemu obnoveniu podľa podmienok zobrazených pri nákupe; už začaté fakturačné obdobie a prípadné vrátenie platieb sa posudzujú podľa objednávky a kogentných právnych predpisov.",
                "Poskytovateľ môže prístup primerane obmedziť alebo pozastaviť pri omeškaní s platbou, porušení týchto podmienok, bezpečnostnom riziku, podozrení na podvod alebo ak je to potrebné na splnenie právnej povinnosti. Ak to okolnosti umožňujú, Poskytovateľ Používateľa vopred upozorní a poskytne mu možnosť nápravy.",
                "Ukončením prístupu zaniká licencia na používanie Softvéru, okrem práv, ktoré musia podľa zákona alebo osobitnej dohody trvať aj po ukončení. Ukončenie nemá vplyv na nároky vzniknuté pred jeho účinnosťou ani na povinnosť vyrovnať splatné záväzky.",
            ],
        },
        {
            "title": "21. Zodpovednosť a obmedzenia",
            "paragraphs": [
                "Každá strana zodpovedá za porušenie svojich zmluvných a zákonných povinností v rozsahu stanovenom príslušnými právnymi predpismi. Poskytovateľ nezodpovedá za obchodné rozhodnutia Používateľa, trhový vývoj, pohyby cien, stratu spôsobenú nastavením Používateľa, konaním Brokera alebo výpadkom služby tretej strany, ak tieto okolnosti nie sú následkom porušenia povinnosti Poskytovateľa.",
                "Poskytovateľ nezaručuje, že Softvér bude bez chýb alebo dostupný bez prerušenia, že AI identifikuje všetky relevantné vzorce alebo že automatizované pokyny budú vykonané Brokerom. Poskytovateľ však zostáva zodpovedný za vlastné povinnosti týkajúce sa poskytovania Služby podľa zmluvy a právnych predpisov.",
                "Žiadne ustanovenie týchto podmienok nevylučuje ani neobmedzuje zodpovednosť alebo právo, ktoré nemožno podľa kogentného právneho predpisu vylúčiť alebo obmedziť, najmä zákonné práva Spotrebiteľa, zodpovednosť za úmyselné konanie alebo iné povinné prípady zodpovednosti.",
            ],
        },
        {
            "title": "22. Sťažnosti a alternatívne riešenie spotrebiteľských sporov",
            "paragraphs": [
                "Otázky, sťažnosti alebo žiadosti o nápravu môže Používateľ zaslať na {{email}}. Poskytovateľ sa pokúsi sťažnosť preskúmať a vyriešiť bez zbytočného odkladu.",
                "Spotrebiteľ má právo obrátiť sa na príslušný subjekt alternatívneho riešenia spotrebiteľských sporov podľa právnych predpisov Slovenskej republiky, ak sú splnené zákonné podmienky. Príslušným subjektom môže byť Slovenská obchodná inšpekcia alebo iný subjekt zo zoznamu Ministerstva hospodárstva Slovenskej republiky podľa povahy sporu. Aktuálne pravidlá a zoznam subjektov je potrebné overiť v čase podania návrhu.",
            ],
        },
        {
            "title": "23. Rozhodné právo a riešenie sporov",
            "paragraphs": [
                "Tieto podmienky a zmluvný vzťah sa riadia právnym poriadkom Slovenskej republiky, pokiaľ kogentné pravidlá medzinárodného práva súkromného neurčujú inak.",
                "Ak je Používateľ Spotrebiteľom s obvyklým pobytom v inom štáte, voľba slovenského práva ho nezbavuje ochrany, ktorú mu poskytujú kogentné ustanovenia práva štátu jeho obvyklého pobytu, ak sa podľa príslušných právnych predpisov uplatnia.",
                "Spor, ktorý sa nepodarí vyriešiť dohodou alebo príslušným mimosúdnym postupom, rozhodne súd s právomocou a príslušnosťou podľa platných právnych predpisov. Tieto podmienky neurčujú súd spôsobom, ktorý by odnímal Spotrebiteľovi zákonné práva.",
            ],
        },
        {
            "title": "24. Zmeny podmienok a záverečné ustanovenia",
            "paragraphs": [
                "Poskytovateľ môže tieto podmienky aktualizovať z dôvodu zmeny Služby, bezpečnosti, technológií alebo právnych predpisov. Nové znenie sa použije od dátumu jeho zverejnenia alebo od neskoršieho oznámeného dátumu. Zmena sa spätne nedotýka už vzniknutých práv a povinností, okrem prípadov, keď to vyžaduje zákon alebo sa strany dohodnú inak.",
                "Ak je niektoré ustanovenie neplatné alebo nevykonateľné, ostatné ustanovenia zostávajú zachované. Neplatné ustanovenie sa nahradí ustanovením, ktoré je platné a čo najviac zodpovedá jeho pôvodnému zákonnému a hospodárskemu účelu.",
                "Slovenské znenie je rozhodujúce, ak sa slovenská a anglická verzia líšia, pokiaľ kogentné právne predpisy nevyžadujú inak.",
                "Posledná aktualizácia: {{date_sk}}",
                "Verzia: {{version}}",
            ],
        },
    ],
    "en": [
        {
            "title": "1. Scope and contractual documents",
            "paragraphs": [
                "These Terms apply to the use of Spectline software, related applications, user accounts, documentation, updates and supported integration features (together, the “Service”).",
                "The contract between the Provider and the User consists of these Terms, the information shown during an order or activation, the order confirmation, the selected plan, any separate agreement and the product documentation. An individually agreed term prevails over these general Terms to the extent of any conflict.",
                "Before completing an order, the User can review the price, feature scope, access period, any recurring billing and other material parameters of the relevant offer. The details confirmed for that order control.",
            ],
        },
        {
            "title": "2. Provider and contact details",
            "paragraphs": [
                "The Service is provided and the contract is entered into by the person identified below. Spectline is the brand used for the Service; the brand name alone does not replace the Provider's legally required business or company identification details.",
            ],
            "bullets": ["{{provider_lines_en}}"],
        },
        {
            "title": "3. Definitions",
            "paragraphs": ["For the purposes of these Terms:"],
            "bullets": [
                "“Provider” means the person identified in Section 2.",
                "“User” means an individual or legal entity that creates an account, orders the Service or uses it.",
                "“Software” or “Spectline” means the software solution, including the AI, analytics, training, testing and integration components made available under the selected plan.",
                "“Trading Data” means historical or current trade, account, price, settings and other data that the User provides or makes accessible to the Software.",
                "“Broker” means a third party that maintains the User's trading account or provides access to a market.",
                "“Trading Platform” means, in particular, MetaTrader 4 or MetaTrader 5 and their relevant technical interfaces.",
                "“Automated Trading” means a feature under which, once enabled, the Software may generate or submit trading instructions through a connected account according to available settings and rules.",
                "“Consumer” means an individual acting outside their trade, business, craft or profession when entering into and performing the contract.",
            ],
        },
        {
            "title": "4. Service and its features",
            "paragraphs": [
                "The Service provides access to Spectline software features within the scope of the purchased or activated plan. Depending on the available version, these may include importing or processing Trading Data, AI-based analysis of trading patterns, strategy training or testing, analytical outputs and integration with MetaTrader 4 or MetaTrader 5.",
                "If the User enables Automated Trading and completes the required connection, the Software may generate and submit trading instructions to the User's account according to selected or confirmed settings. The User acknowledges that such instructions may be submitted without separate manual confirmation of each individual trade.",
                "The exact feature scope, supported platforms, technical requirements, available markets and any limits are determined by the current product documentation and the offer for the relevant plan. A feature not expressly listed in the plan or documentation is not part of the Service.",
            ],
        },
        {
            "title": "5. Regulatory nature and limits of the Service",
            "paragraphs": [
                "These Terms govern the provision of and access to Software. Describing a product as software or a technical tool does not, by itself, determine the legal classification of its actual features. The actual operation, degree of discretion, permissions over the User's account and the way trading instructions are generated and executed are relevant.",
                "Under these Terms, the Provider does not undertake to provide personal investment recommendations based on the User's individual financial circumstances, and these Terms do not constitute a portfolio-management agreement or an agreement to execute client orders. However, no wording in these Terms changes the legal nature of an activity if a particular feature or operating model constitutes a regulated investment service or activity under mandatory law.",
                "A feature that constitutes a regulated investment service or activity may only be provided where and by an entity that has the required authorisation or can rely on an applicable statutory exemption. Where necessary, it must be covered by a separate agreement and the relevant regulatory documentation. These Terms are not an authorisation to provide investment services.",
                "The User must not treat analytical outputs, signals, scores, back-tests or AI behaviour as personal investment advice or as a guarantee that a particular trade is suitable for the User.",
            ],
        },
        {
            "title": "6. Trading risk warning",
            "paragraphs": [
                "Trading financial instruments, currencies, CFDs, derivatives or other products available through a Broker involves risk. The risk is particularly high for leveraged products. The account value may fall rapidly and the User may lose part or all of the capital allocated to trading. Depending on the product, the Broker agreement and applicable protections, losses may in certain circumstances exceed the amount originally allocated.",
                "The User trades at their own risk and should trade only with funds they can afford to lose. Before using a live account, the User should thoroughly test the Software and its settings on a demo account, understand the products being traded and independently assess the strategy's risk.",
                "No strategy, algorithm, AI model, historical result or back-test eliminates market risk. Stop Loss, Take Profit or other protective settings may not execute at the expected price or time, for example due to price gaps, low liquidity, volatility or a connection failure.",
                "The User should not enable Automated Trading unless they understand how it works, its risk parameters and how to stop trading.",
            ],
        },
        {
            "title": "7. AI analysis, historical data and performance",
            "paragraphs": [
                "AI features analyse Trading Data and may identify statistical patterns or generate outputs based on available data, models and settings. Results depend on the quality, completeness, representativeness and accuracy of input data, as well as the methodology and market conditions.",
                "Historical performance, back-testing, simulations, training results and results achieved by other users are not reliable indicators of future performance. Simulation results may differ materially from live trading due to spreads, commissions, swaps, slippage, latency, liquidity, price availability, outages or changes in market conditions.",
                "The Provider does not guarantee profit, a particular success rate, prediction accuracy, number of trades, minimum return, passive income or that the Software will prevent losses. Any example, chart or historical result is informational or illustrative only unless expressly and lawfully agreed otherwise.",
                "AI models and software outputs may contain errors, bias, omissions or inaccuracies. The Provider may change models and algorithms while maintaining the agreed features of the Service and the User's applicable legal rights.",
            ],
        },
        {
            "title": "8. User account and eligibility",
            "paragraphs": [
                "The User must provide true, complete and up-to-date information during registration and use of the Service and keep that information current. The Service may only be used by a person with legal capacity to enter into a contract and who is eligible to use the relevant trading and brokerage services in their jurisdiction.",
                "The User is responsible for protecting login details, passwords, access tokens and other authentication credentials. The User must promptly notify the Provider of suspected unauthorised access or a security incident.",
                "The User must not make the account available to another person, circumvent security or licence restrictions, or use the Service in a manner that breaches these Terms or third-party rights.",
            ],
        },
        {
            "title": "9. Brokers and MetaTrader 4 / MetaTrader 5 integration",
            "paragraphs": [
                "Some features may require a separate trading account with a Broker, MetaTrader 4 or MetaTrader 5 and a suitable internet connection. The User's Broker account, funding, trading terms, fees, margin requirements, order execution and client protections are governed by a separate agreement between the User and the Broker.",
                "The Provider is not a party to the relationship between the User and the Broker and is not responsible for the Broker's decisions, terms, availability, rejection or delay of an order, execution price, outages or changes to its platform, except for liability that cannot be excluded under applicable law.",
                "MetaTrader, MetaQuotes and Broker names are identifiers of their respective owners. Their mention is for the purpose of identifying compatible platforms and does not by itself imply ownership, approval or partnership with the Provider.",
                "Compatibility may vary depending on the platform version, Broker, account type, operating system, server and technical interfaces. The Provider does not guarantee support for every configuration or future third-party version.",
            ],
        },
        {
            "title": "10. User's duties and responsibility for trading",
            "paragraphs": [
                "The User is responsible for selecting the Broker and traded product, checking eligibility to trade, configuring the account, capital, leverage, position sizes, risk limits and deciding whether to enable Automated Trading.",
                "Before enabling Automated Trading, the User must review the strategy parameters, supported symbols and timeframes, position size, Stop Loss, Take Profit, maximum acceptable loss and any other settings available in the product. The User must reasonably monitor the Software's operation and know how to pause or disable Automated Trading.",
                "The User must not use the Service for market manipulation, fraud, money laundering, sanctions evasion, breach of Broker rules, trading rules or applicable law. The User is responsible for their tax and regulatory obligations relating to trading.",
                "The Provider is not required to monitor the User's trading account, notify the User about every trade or intervene in the strategy unless such a duty is expressly agreed in a separate agreement and is legally permissible.",
            ],
        },
        {
            "title": "11. Trading Data, User inputs and processing permission",
            "paragraphs": [
                "The User is responsible for having the right to upload, make available and use Trading Data and other materials in the Service for the purposes agreed in these Terms. The User must not provide data whose processing would breach another person's rights, the Broker agreement or applicable law.",
                "The User grants the Provider a non-exclusive, worldwide permission, to the extent necessary to operate the Service, to store, copy, transmit, analyse and technically modify the supplied data for the purposes of providing features, support, security and performing the contract. This permission does not transfer ownership of the data to the Provider.",
                "Any use of data for separate purposes beyond providing the Service, in particular training general or shared models, marketing or disclosure to third parties, must be transparently disclosed to the User and must have a separate valid legal basis or consent where required by law.",
                "The User must not upload unnecessary personal data of third parties, credentials that enable unauthorised handling of funds, or malicious code.",
            ],
        },
        {
            "title": "12. Availability, maintenance, updates and technical limits",
            "paragraphs": [
                "The Provider will use reasonable efforts to operate and maintain the Service. However, it does not guarantee uninterrupted availability, error-free operation, immediate response or compatibility with all devices, networks, systems, data sources or third-party services.",
                "The Service may be temporarily restricted or unavailable due to maintenance, updates, security measures, internet, cloud, data-source, Broker or Trading Platform outages or another event outside the Provider's reasonable control. Where practicable, the Provider will give advance notice of planned maintenance.",
                "The Provider may release fixes, updates and feature changes. For digital content or digital services supplied to a Consumer, updates and changes will be made in accordance with mandatory legal requirements concerning conformity of digital content or services and consumer information.",
            ],
        },
        {
            "title": "13. Price, payments and any subscription",
            "paragraphs": [
                "The User must pay the price stated in the order, quotation or order confirmation. The price, currency, included features, billing period, taxes and any additional fees must be shown to the User before the binding order is submitted.",
                "Unless the invoice or an individual agreement states otherwise, the invoice payment period is {{due_days}} days from its issue date. The Provider will issue an accounting document in accordance with applicable law.",
                "If a plan operates as a subscription or involves recurring payments, its price, billing-period length, renewal date and method, price-change conditions and cancellation procedure will be stated before the order is placed. Recurring billing will apply only where it was clearly disclosed in the order and accepted by the User.",
                "If payment is overdue, the Provider may, after prior notice, reasonably restrict or suspend access to the paid Service where permitted by law and without affecting mandatory Consumer rights. Statutory default interest may be claimed only to the extent permitted by law.",
            ],
        },
        {
            "title": "14. Licence and intellectual property",
            "paragraphs": [
                "Unless the order states otherwise, the Provider grants the User a non-exclusive, non-transferable and non-sublicensable licence, for the paid and agreed access period, to use the Software for the User's own internal purposes and within the features of the relevant plan.",
                "All rights in the Software, source code, models, algorithms, interfaces, documentation, design, brand, databases and related materials remain with the Provider or the relevant third parties. The User receives only the rights expressly granted by these Terms or a separate licence agreement.",
                "The User must not copy, sell, rent, sublicense or make the Software available as a service to a third party, circumvent technical protections, reverse engineer or decompile it, except to the extent that such restriction cannot be imposed under mandatory law.",
                "Rights in MetaTrader, data sources, libraries, plugins and other third-party components are governed by their own licence terms.",
            ],
        },
        {
            "title": "15. Prohibited use and security",
            "paragraphs": [
                "The User must not use the Service for unlawful purposes, disrupt its operation, attempt unauthorised access, distribute malicious code, circumvent limits or security measures, interfere with other users' accounts or overload the infrastructure.",
                "The Provider may take reasonable security measures, restrict access or suspend an account if it has reasonable grounds to suspect a security incident, fraud, breach of law or breach of these Terms. Where circumstances permit, the Provider will inform the User of the reason.",
            ],
        },
        {
            "title": "16. Third parties and external services",
            "paragraphs": [
                "The Service may depend on infrastructure, cloud providers, data sources, payment services, Brokers, Trading Platforms or other external services. Their availability and use are governed by the relevant third party's terms.",
                "The Provider is not responsible for the content, terms, security or availability of a separate third-party service to the extent that it is outside the Provider's control. This does not affect the Provider's liability for its own breach of contractual or legal duties.",
            ],
        },
        {
            "title": "17. Consumer rights, withdrawal and digital content or services",
            "paragraphs": [
                "Where the User is a Consumer and the contract was concluded at a distance or away from the Provider's business premises, the Consumer has a right to withdraw within the period and under the conditions set by law, generally within 14 days after entering into a service contract or a contract for the supply of digital content. Applicable exceptions and the start of the period depend on the legal nature of the particular performance.",
                "If the supply of digital content or provision of a digital service is to begin before the withdrawal period expires, the Provider will obtain the Consumer's express request or consent and provide the notices and confirmations required by law. This clause does not replace a separate consent or acknowledgement that must be obtained in the checkout process.",
                "The right of withdrawal may be lost or restricted only in cases and under the conditions expressly provided by law, for example for digital content after the statutory requirements for express consent, acknowledgement and confirmation have been satisfied. Different statutory rules apply to services. All mandatory Consumer rights remain unaffected.",
                "A withdrawal notice may be sent to {{email}} or made by another unequivocal statement delivered to the Provider. The Consumer may also use a model withdrawal form if one has been provided.",
                "For digital content or services supplied to a Consumer, the Consumer has statutory rights relating to conformity with the contract, functionality, compatibility, updates and remedies. These rights may not be excluded or restricted by these Terms.",
            ],
        },
        {
            "title": "18. Complaints and remedies for non-conforming performance",
            "paragraphs": [
                "The User may report a technical problem, outage or failure of the Service to match its agreed description to {{email}}. The report should include the account identifier, description of the issue, when it occurred and available technical information. The User should not send passwords or full Broker account credentials.",
                "The Provider will review the report and, depending on the issue, propose an appropriate remedy, such as correction, restoration of access, re-supply of digital content or another remedy under the contract and applicable law.",
                "For Consumers, statutory rules on defects and conformity of digital content or services apply, including statutory periods and remedies. These Terms do not restrict Consumer rights. For Users who are not Consumers, defect rights are assessed under the contract and applicable law.",
                "A trading loss, an unfavourable strategy result or a difference between live trading and a historical test is not automatically a technical defect in the Software. This does not affect the User's rights where the issue resulted from the Provider's breach or a defect in the Service.",
            ],
        },
        {
            "title": "19. Personal data and confidentiality",
            "paragraphs": [
                "Personal data processing is governed by the separate Privacy Policy available at {{website}}. That notice should explain, in particular, the purposes, legal bases, categories of data, recipients, retention periods and data-subject rights.",
                "Where the Provider processes personal data on behalf of the User as a processor, the parties will enter into a data processing agreement under Article 28 GDPR where such an agreement is required by the actual processing arrangement.",
                "The parties will protect non-public confidential information obtained through use of the Service and use it only to perform the contract, comply with legal obligations or protect their rights. This duty does not apply to information that is public without breach of an obligation or must be disclosed by law.",
            ],
        },
        {
            "title": "20. Suspension and termination of access",
            "paragraphs": [
                "The User may stop using the Service and cancel a subscription using the procedure stated in the order or user account. For a subscription, cancellation will prevent further renewal in accordance with the conditions shown at purchase; an already started billing period and any refund are governed by the order and mandatory law.",
                "The Provider may reasonably restrict or suspend access for overdue payment, breach of these Terms, a security risk, suspected fraud or where necessary to comply with a legal obligation. Where circumstances permit, the Provider will notify the User in advance and allow an opportunity to remedy the issue.",
                "When access ends, the licence to use the Software ends, except for rights that must continue under law or a separate agreement. Termination does not affect claims accrued before its effective date or the obligation to settle amounts due.",
            ],
        },
        {
            "title": "21. Liability and limitations",
            "paragraphs": [
                "Each party is liable for breach of its contractual and legal duties to the extent provided by applicable law. The Provider is not responsible for the User's trading decisions, market movements, price changes, losses caused by the User's settings, Broker conduct or third-party service outages, unless those circumstances result from the Provider's breach of duty.",
                "The Provider does not warrant that the Software will be error-free or uninterrupted, that AI will identify every relevant pattern or that the Broker will execute automated instructions. The Provider remains liable for its own obligations to provide the Service under the contract and applicable law.",
                "Nothing in these Terms excludes or limits liability or a right that cannot be excluded or limited under mandatory law, in particular statutory Consumer rights, liability for intentional conduct or any other mandatory liability.",
            ],
        },
        {
            "title": "22. Complaints and alternative consumer dispute resolution",
            "paragraphs": [
                "Questions, complaints or requests for redress may be sent to {{email}}. The Provider will seek to review and resolve the complaint without undue delay.",
                "A Consumer may apply to a competent alternative dispute resolution entity under Slovak law where the statutory conditions are met. Depending on the nature of the dispute, the competent entity may be the Slovak Trade Inspection or another entity on the list maintained by the Slovak Ministry of Economy. The applicable rules and list should be checked at the time the application is submitted.",
            ],
        },
        {
            "title": "23. Governing law and disputes",
            "paragraphs": [
                "These Terms and the contractual relationship are governed by the laws of the Slovak Republic unless mandatory conflict-of-law rules provide otherwise.",
                "Where the User is a Consumer habitually resident in another country, the choice of Slovak law does not deprive the Consumer of the protection afforded by mandatory provisions of the law of their habitual residence where those provisions apply.",
                "A dispute that cannot be resolved amicably or through an applicable out-of-court procedure will be decided by a court having jurisdiction under applicable law. These Terms do not determine jurisdiction in a way that removes statutory Consumer rights.",
            ],
        },
        {
            "title": "24. Amendments and final provisions",
            "paragraphs": [
                "The Provider may update these Terms due to changes to the Service, security, technology or applicable law. The revised wording will apply from the date of publication or a later notified date. A change does not retrospectively affect rights and obligations already accrued, except where required by law or otherwise agreed by the parties.",
                "If any provision is invalid or unenforceable, the remaining provisions remain in effect. The invalid provision will be replaced with a valid provision that most closely reflects its original lawful and commercial purpose.",
                "The Slovak version prevails if the Slovak and English versions differ, unless mandatory law requires otherwise.",
                "Last updated: {{date_en}}",
                "Version: {{version}}",
            ],
        },
    ],
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
        if (
            key == "brand_name"
            and value.casefold() == str(provider.get("legal_name", "")).strip().casefold()
        ):
            continue
        lines.append(f"{labels[key]}: {value}")
    return lines


def build_sections(language: str) -> list[dict]:
    if language not in SECTIONS:
        raise ValueError(f"Unsupported language: {language}")

    replacements = {
        "{{email}}": DATA["provider"]["email"],
        "{{website}}": DATA["website"],
        "{{due_days}}": str(DATA["commercial_defaults"]["payment_due_days"]),
        "{{date_sk}}": formatted_date("sk"),
        "{{date_en}}": formatted_date("en"),
        "{{version}}": DATA["version"],
        "{{provider_lines_sk}}": "\n".join(provider_lines("sk")),
        "{{provider_lines_en}}": "\n".join(provider_lines("en")),
    }

    result: list[dict] = []
    for source_section in SECTIONS[language]:
        section = {
            "title": source_section["title"],
            "paragraphs": [
                _replace_tokens(value, replacements)
                for value in source_section.get("paragraphs", [])
            ],
            "bullets": [
                _replace_tokens(value, replacements)
                for value in source_section.get("bullets", [])
            ],
        }
        result.append(section)
    return result


def _replace_tokens(value: str, replacements: dict[str, str]) -> str:
    for token, replacement in replacements.items():
        value = value.replace(token, replacement)
    return value


def ensure_paths() -> None:
    if not os.path.isfile(FONT_PATH):
        raise FileNotFoundError(
            f"Missing font: {FONT_PATH}\n"
            "Expected location: <project>/public/fonts/Roboto-Regular.ttf"
        )
    if not os.path.isfile(LOGO_PATH):
        raise FileNotFoundError(
            f"Missing logo: {LOGO_PATH}\n"
            "Expected location: <project>/public/images/logo_black.png"
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
            fontSize=9.2,
            leading=12.4,
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
    canv.setFont("Roboto", 15)
    canv.setFillColor(ACCENT)
    canv.drawRightString(width - 2 * cm, height - 2.2 * cm, title)

    canv.setFont("Roboto", 8)
    canv.setFillColor(MUTED)
    canv.drawString(2 * cm, 1.2 * cm, f"{DATA['website']}  |  {formatted_date(language)}")
    canv.drawRightString(width - 2 * cm, 1.2 * cm, f"{labels['page']} {doc.page}")

    canv.setTitle(title)
    canv.setAuthor(DATA["provider"]["brand_name"])
    canv.setSubject("Software Terms and Conditions" if language == "en" else "Podmienky používania softvéru")
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
        subject="Software Terms and Conditions" if language == "en" else "Podmienky používania softvéru",
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

    page_callback = partial(draw_page, title=labels["title"], language=language)
    document.build(story, onFirstPage=page_callback, onLaterPages=page_callback)


def main() -> None:
    ensure_paths()
    register_fonts()
    generate_pdf("sk", OUTPUT_PDF_SK)
    generate_pdf("en", OUTPUT_PDF_EN)
    print(f"Generated: {OUTPUT_PDF_SK}")
    print(f"Generated: {OUTPUT_PDF_EN}")


if __name__ == "__main__":
    main()
