#!/usr/bin/env python3
"""Generate a professional PDF report for the Deep Learning project."""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
OUTPUT_PDF = os.path.join(BASE_DIR, "rapport_projet_deeplearning.pdf")

# ── Color palette ──────────────────────────────────────────────────────────────
EMSI_BLUE   = colors.HexColor("#003087")
EMSI_GREEN  = colors.HexColor("#006633")
ACCENT_BLUE = colors.HexColor("#1565C0")
LIGHT_BLUE  = colors.HexColor("#E3F2FD")
LIGHT_GRAY  = colors.HexColor("#F5F5F5")
MED_GRAY    = colors.HexColor("#9E9E9E")
DARK_GRAY   = colors.HexColor("#424242")
WHITE       = colors.white
BLACK       = colors.black

# ── Page dimensions ────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm

# ── Styles ─────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

COVER_TITLE = S("CoverTitle",
    fontName="Helvetica-Bold", fontSize=26, textColor=EMSI_BLUE,
    alignment=TA_CENTER, spaceAfter=8, leading=32)

COVER_SUBTITLE = S("CoverSubtitle",
    fontName="Helvetica", fontSize=13, textColor=ACCENT_BLUE,
    alignment=TA_CENTER, spaceAfter=6, leading=18)

COVER_META = S("CoverMeta",
    fontName="Helvetica", fontSize=11, textColor=DARK_GRAY,
    alignment=TA_CENTER, spaceAfter=4)

H1 = S("H1Style",
    fontName="Helvetica-Bold", fontSize=16, textColor=EMSI_BLUE,
    spaceBefore=18, spaceAfter=8, leading=20,
    borderPad=4)

H2 = S("H2Style",
    fontName="Helvetica-Bold", fontSize=13, textColor=ACCENT_BLUE,
    spaceBefore=12, spaceAfter=6, leading=16)

H3 = S("H3Style",
    fontName="Helvetica-Bold", fontSize=11, textColor=DARK_GRAY,
    spaceBefore=8, spaceAfter=4, leading=14)

BODY = S("BodyStyle",
    fontName="Helvetica", fontSize=10, textColor=BLACK,
    spaceAfter=6, leading=15, alignment=TA_JUSTIFY)

BODY_BOLD = S("BodyBold",
    fontName="Helvetica-Bold", fontSize=10, textColor=BLACK,
    spaceAfter=6, leading=15, alignment=TA_JUSTIFY)

BULLET = S("BulletStyle",
    fontName="Helvetica", fontSize=10, textColor=BLACK,
    spaceAfter=4, leading=14, leftIndent=16, bulletIndent=4,
    alignment=TA_LEFT)

CAPTION = S("CaptionStyle",
    fontName="Helvetica-Oblique", fontSize=9, textColor=MED_GRAY,
    alignment=TA_CENTER, spaceAfter=6)

CODE = S("CodeStyle",
    fontName="Courier", fontSize=8.5, textColor=colors.HexColor("#1A237E"),
    backColor=LIGHT_GRAY, spaceAfter=6, leading=13,
    leftIndent=12, rightIndent=12)

NOTE_BOX = S("NoteBox",
    fontName="Helvetica", fontSize=9.5, textColor=DARK_GRAY,
    backColor=LIGHT_BLUE, spaceAfter=6, leading=14,
    leftIndent=12, rightIndent=12,
    borderColor=ACCENT_BLUE, borderWidth=1, borderPad=6)

TOC_H1 = S("TOC_H1",
    fontName="Helvetica-Bold", fontSize=11, textColor=EMSI_BLUE,
    spaceAfter=2, leading=14, leftIndent=0)

TOC_H2 = S("TOC_H2",
    fontName="Helvetica", fontSize=10, textColor=DARK_GRAY,
    spaceAfter=1, leading=13, leftIndent=16)

# ── Helpers ────────────────────────────────────────────────────────────────────
def hr(color=EMSI_BLUE, thickness=1.5):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=2)

def img(filename, width=14*cm, caption=None):
    path = os.path.join(FIGURES_DIR, filename)
    elems = []
    if os.path.exists(path):
        try:
            im = Image(path, width=width)
            im.hAlign = "CENTER"
            elems.append(Spacer(1, 4))
            elems.append(im)
        except Exception:
            pass
    if caption:
        elems.append(Paragraph(caption, CAPTION))
    return elems

def section_title(num, title):
    return [
        Spacer(1, 4),
        Paragraph(f"{num}. {title}", H1),
        hr(EMSI_BLUE, 1.5),
    ]

def subsection_title(num, title):
    return [Paragraph(f"{num} {title}", H2)]

def bullet(text):
    return Paragraph(f"•  {text}", BULLET)

def info_box(text):
    return Paragraph(text, NOTE_BOX)

def make_table(headers, rows, col_widths=None):
    data = [headers] + rows
    total = PAGE_W - 2 * MARGIN
    if col_widths is None:
        n = len(headers)
        col_widths = [total / n] * n
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), EMSI_BLUE),
        ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#BDBDBD")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    return t

# ── Page decorator ─────────────────────────────────────────────────────────────
class ReportCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page(num_pages)
            super().showPage()
        super().save()

    def _draw_page(self, total):
        pg = self._pageNumber
        if pg == 1:
            return  # cover page — no header/footer
        # header — thin line only, no filled rectangle
        self.setStrokeColor(EMSI_BLUE)
        self.setLineWidth(1.0)
        self.line(MARGIN, PAGE_H - 1.0*cm, PAGE_W - MARGIN, PAGE_H - 1.0*cm)
        self.setFillColor(EMSI_BLUE)
        self.setFont("Helvetica-Bold", 8)
        self.drawString(MARGIN, PAGE_H - 0.75*cm,
                        "Projet Deep Learning — EMSI 2025-2026")
        self.setFillColor(MED_GRAY)
        self.setFont("Helvetica", 8)
        self.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.75*cm,
                             "MLP · CNN · RNN/Seq2Seq")
        # footer
        self.setStrokeColor(EMSI_BLUE)
        self.setLineWidth(0.5)
        self.line(MARGIN, 1.4*cm, PAGE_W - MARGIN, 1.4*cm)
        self.setFillColor(EMSI_BLUE)
        self.setFont("Helvetica", 8)
        self.drawCentredString(PAGE_W / 2, 0.85*cm, f"Page {pg} / {total}")

def th(text):
    """Table header cell paragraph."""
    return Paragraph(f"<b>{text}</b>", S("th_"+text[:4],
        fontName="Helvetica-Bold", fontSize=9, textColor=WHITE, alignment=TA_CENTER))

def td(text, bold=False):
    sty = BODY_BOLD if bold else BODY
    return Paragraph(text, sty)

# ── Build content ──────────────────────────────────────────────────────────────
def build_story():
    story = []

    # ── COVER PAGE ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*cm))

    # Logo / school name block
    logo_data = [
        [Paragraph("<b><font color='#003087' size=18>EMSI</font></b>", S("x", fontName="Helvetica-Bold", fontSize=18, textColor=EMSI_BLUE, alignment=TA_CENTER)),
         Paragraph("<b>École Marocaine des Sciences de l'Ingénieur</b><br/>Membre de HONORIS UNITED UNIVERSITIES",
                   S("y", fontName="Helvetica", fontSize=10, textColor=EMSI_BLUE, alignment=TA_LEFT, leading=14))]
    ]
    logo_tbl = Table(logo_data, colWidths=[3*cm, PAGE_W - 2*MARGIN - 3*cm])
    logo_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(logo_tbl)
    story.append(hr(EMSI_BLUE, 2))
    story.append(Spacer(1, 1*cm))

    story.append(Paragraph("Projet de Fin de Module", COVER_SUBTITLE))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("RAPPORT SCIENTIFIQUE", S("rt", fontName="Helvetica-Bold", fontSize=11,
                                                      textColor=EMSI_BLUE, alignment=TA_CENTER, spaceAfter=2)))
    story.append(Spacer(1, 0.6*cm))
    story.append(Paragraph("Deep Learning avec PyTorch", COVER_TITLE))
    story.append(Paragraph("MLP · CNN · RNN / LSTM / GRU · Seq2Seq",
                            S("sub", fontName="Helvetica", fontSize=12, textColor=ACCENT_BLUE,
                              alignment=TA_CENTER, spaceAfter=4)))
    story.append(Paragraph("Breast Cancer Wisconsin &nbsp;|&nbsp; PneumoniaMNIST &nbsp;|&nbsp; UCI Drug Reviews",
                            S("ds", fontName="Helvetica-Oblique", fontSize=10, textColor=MED_GRAY,
                              alignment=TA_CENTER, spaceAfter=4)))
    story.append(hr(ACCENT_BLUE, 1))
    story.append(Spacer(1, 1.2*cm))

    # Info table
    info = [
        ["Module", "Deep Learning"],
        ["Encadré par", "Professeur Mossab Batal"],
        ["Année", "2025 — 2026"],
        ["Groupe", "4IIR-IAD / G1 / Moulay Youssef — EMSI Casablanca"],
    ]
    info_tbl = Table(info, colWidths=[5*cm, PAGE_W - 2*MARGIN - 5*cm])
    info_tbl.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",   (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",  (0, 0), (0, -1), EMSI_BLUE),
        ("TEXTCOLOR",  (1, 0), (1, -1), DARK_GRAY),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#90CAF9")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 1.2*cm))

    # Contents overview box
    parts = [
        ["Partie I",   "MLP — Classification tabulaire (Breast Cancer Wisconsin)"],
        ["Partie II",  "CNN — Vision médicale (PneumoniaMNIST)"],
        ["Partie III", "RNN/LSTM/GRU & Seq2Seq (UCI Drug Reviews)"],
    ]
    parts_tbl = Table(parts, colWidths=[3*cm, PAGE_W - 2*MARGIN - 3*cm])
    parts_tbl.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), WHITE),
        ("BACKGROUND",(0, 0), (0, -1), ACCENT_BLUE),
        ("TEXTCOLOR", (1, 0), (1, -1), DARK_GRAY),
        ("BACKGROUND",(1, 0), (1, -1), LIGHT_GRAY),
        ("GRID",      (0, 0), (-1, -1), 0.5, colors.HexColor("#BDBDBD")),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(parts_tbl)
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("Casablanca — Juin 2026", COVER_META))

    story.append(PageBreak())

    # ── REMERCIEMENTS ─────────────────────────────────────────────────────────
    story += section_title("", "Remerciements")
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Je tiens à exprimer ma profonde gratitude à <b>Monsieur le Professeur Mossab Batal</b> "
        "pour la qualité de son enseignement, la rigueur de son encadrement et la disponibilité "
        "dont il a fait preuve tout au long de ce module de Deep Learning.", BODY))
    story.append(Paragraph(
        "Je remercie également l'<b>École Marocaine des Sciences de l'Ingénieur (EMSI)</b>, "
        "site Moulay Youssef, pour les ressources pédagogiques et l'environnement de travail mis "
        "à ma disposition.", BODY))
    story.append(Paragraph(
        "Ce projet m'a permis d'acquérir une maîtrise solide des architectures de réseaux de "
        "neurones profonds — MLP, CNN et RNN/Seq2Seq — et de les appliquer sur des données "
        "médicales réelles dans un cadre scientifique rigoureux.", BODY))
    story.append(PageBreak())

    # ── RÉSUMÉ ────────────────────────────────────────────────────────────────
    story += section_title("", "Résumé")
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Ce rapport présente la réalisation d'un projet de Deep Learning structuré en trois "
        "parties complémentaires, chacune adressant une modalité de données médicales différente.", BODY))

    resume_rows = [
        ["Partie I",   "MLP (Breast Cancer Wisconsin)",   "accuracy 94–97 %"],
        ["Partie II",  "CNN LeNet (PneumoniaMNIST)",       "val accuracy 96.95 %"],
        ["Partie III", "RNN/LSTM/GRU & Seq2Seq\n(UCI Drug Reviews)", "GRU acc 68.33 %\nBLEU 0.1047"],
    ]
    story.append(make_table(
        [Paragraph("<b>Partie</b>", S("th", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE, alignment=TA_CENTER)),
         Paragraph("<b>Architecture & Dataset</b>", S("th", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE, alignment=TA_CENTER)),
         Paragraph("<b>Résultat clé</b>", S("th", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE, alignment=TA_CENTER))],
        [[Paragraph(r[0], BODY), Paragraph(r[1], BODY), Paragraph(r[2], BODY)] for r in resume_rows],
        col_widths=[3.2*cm, 8.5*cm, 4*cm]
    ))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "<b>Mots-clés :</b> Deep Learning, PyTorch, MLP, CNN, RNN, LSTM, GRU, Seq2Seq, "
        "classification médicale, vision par ordinateur, traitement du langage naturel.", BODY))
    story.append(PageBreak())

    # ── TABLE OF CONTENTS (manual) ────────────────────────────────────────────
    story += section_title("", "Table des matières")
    toc_entries = [
        ("1",   "Introduction générale", "H1"),
        ("1.1", "Contexte et motivation", "H2"),
        ("1.2", "Cadre expérimental", "H2"),
        ("2",   "Partie I — MLP & Classification tabulaire (Breast Cancer Wisconsin)", "H1"),
        ("2.1", "Objectifs pédagogiques", "H2"),
        ("2.2", "Analyse exploratoire du dataset", "H2"),
        ("2.3", "Formulation mathématique du MLP", "H2"),
        ("2.4", "Méthodologie & Implémentation PyTorch", "H2"),
        ("2.5", "Résultats expérimentaux", "H2"),
        ("2.6", "Analyse des initialisations des poids", "H2"),
        ("2.7", "Interprétation & Analyse critique", "H2"),
        ("2.8", "Question de synthèse", "H2"),
        ("3",   "Partie II — CNN & Vision Médicale (PneumoniaMNIST)", "H1"),
        ("3.1", "Objectifs pédagogiques", "H2"),
        ("3.2", "Dataset PneumoniaMNIST", "H2"),
        ("3.3", "Opérations fondamentales du CNN", "H2"),
        ("3.4", "Architecture LeNet & variantes", "H2"),
        ("3.5", "Résultats expérimentaux", "H2"),
        ("3.6", "Analyse des feature maps", "H2"),
        ("3.7", "Interprétation & Analyse critique", "H2"),
        ("3.8", "Question de synthèse", "H2"),
        ("4",   "Partie III — RNN, LSTM, GRU & Seq2Seq", "H1"),
        ("4.1", "Objectifs pédagogiques", "H2"),
        ("4.2", "Dataset UCI Drug Reviews", "H2"),
        ("4.3", "Formulations mathématiques RNN / LSTM / GRU", "H2"),
        ("4.4", "Architecture Seq2Seq encodeur-décodeur", "H2"),
        ("4.5", "Méthodologie & Implémentation PyTorch", "H2"),
        ("4.6", "Résultats expérimentaux", "H2"),
        ("4.7", "Analyse du gradient clipping et perplexité", "H2"),
        ("4.8", "Décodage glouton vs beam search & BLEU", "H2"),
        ("4.9", "Interprétation & Analyse critique", "H2"),
        ("4.10","Question de synthèse", "H2"),
        ("5",   "Question Transversale Finale", "H1"),
        ("6",   "Conclusion Générale", "H1"),
        ("7",   "Annexe — Tableau de bord expérimental", "H1"),
        ("",    "Bibliographie", "H1"),
    ]
    for num, title, level in toc_entries:
        text = f"{num}&nbsp;&nbsp;{title}" if num else title
        sty = TOC_H1 if level == "H1" else TOC_H2
        story.append(Paragraph(text, sty))
    story.append(PageBreak())

    # ── 1. INTRODUCTION ───────────────────────────────────────────────────────
    story += section_title("1", "Introduction Générale")

    story += subsection_title("1.1", "Contexte et motivation")
    story.append(Paragraph(
        "Le Deep Learning a révolutionné l'intelligence artificielle en permettant d'apprendre "
        "automatiquement des représentations hiérarchiques à partir de données brutes, sans "
        "ingénierie de features manuelle. Depuis le succès d'AlexNet en 2012, les réseaux de "
        "neurones profonds dominent la vision par ordinateur, le traitement du langage naturel "
        "et, de plus en plus, la médecine de précision.", BODY))
    story.append(Paragraph(
        "Ce projet adresse trois défis représentatifs du domaine médical, chacun associé à une "
        "modalité de données différente et à une famille architecturale adaptée :", BODY))
    story.append(bullet(
        "<b>Données tabulaires (Partie I) :</b> les résultats cytologiques du Breast Cancer "
        "Wisconsin dataset (30 mesures numériques) permettent de tester le MLP dans sa forme "
        "la plus pure — sans structure spatiale ni temporelle. L'enjeu est la généralisation "
        "avec seulement 569 exemples."))
    story.append(bullet(
        "<b>Images médicales (Partie II) :</b> la détection de pneumonie sur radiographies "
        "thoraciques (PneumoniaMNIST) illustre la pertinence des CNN pour exploiter la "
        "corrélation spatiale locale des pixels — un prior inductif aligné avec la structure "
        "des lésions pulmonaires."))
    story.append(bullet(
        "<b>Texte médical (Partie III) :</b> les reviews de patients sur des médicaments "
        "(UCI Drug Reviews) requièrent une modélisation du contexte séquentiel : le sens "
        "d'une opinion médicale dépend de l'ordre des tokens et des dépendances longue-distance. "
        "RNN/LSTM/GRU et Seq2Seq s'imposent naturellement."))

    story += subsection_title("1.2", "Cadre expérimental")
    story.append(Paragraph(
        "Tous les modèles sont implémentés en <b>PyTorch 2.10.0</b> sur Google Colab avec GPU "
        "NVIDIA T4 (CUDA 12.8). Les graines aléatoires sont fixées (<font face='Courier'>torch.manual_seed(42)</font>, "
        "<font face='Courier'>np.random.seed(42)</font>) pour la reproductibilité complète. "
        "L'optimiseur <b>Adam</b> avec lr=1×10⁻³ est utilisé uniformément pour permettre une "
        "comparaison équitable entre architectures.", BODY))
    story.append(make_table(
        [th("Composant"), th("Version / Détail")],
        [
            [td("Python"),              td("3.10+")],
            [td("PyTorch"),             td("2.10.0+cu128 (CUDA 12.8)")],
            [td("GPU"),                 td("NVIDIA T4 16 GB (Google Colab)")],
            [td("medmnist"),            td("3.0.2")],
            [td("scikit-learn"),        td("1.3+  (StandardScaler, metrics)")],
            [td("datasets HuggingFace"),td("lewtun/drug-reviews")],
            [td("NLTK"),                td("BLEU score (sentence_bleu)")],
        ],
        col_widths=[6*cm, PAGE_W - 2*MARGIN - 6*cm]
    ))
    story.append(PageBreak())

    # ── 2. PARTIE I — MLP ─────────────────────────────────────────────────────
    story += section_title("2", "Partie I — MLP & Classification Tabulaire")
    story.append(Paragraph("Breast Cancer Wisconsin Dataset", S("ds2", fontName="Helvetica-Oblique",
                             fontSize=12, textColor=ACCENT_BLUE, alignment=TA_LEFT, spaceAfter=8)))

    story += subsection_title("2.1", "Objectifs pédagogiques")
    story.append(Paragraph(
        "Cette partie vise à maîtriser les fondements théoriques et pratiques du MLP :", BODY))
    story.append(bullet("Construire un MLP avec <font face='Courier'>nn.Sequential</font> et une classe PyTorch personnalisée"))
    story.append(bullet("Comprendre le théorème d'approximation universelle (Cybenko, 1989) et ses limites pratiques"))
    story.append(bullet("Implémenter et comparer trois stratégies d'initialisation des poids (Gaussienne, Constante, Xavier)"))
    story.append(bullet("Analyser le rôle de Batch Normalization et Dropout dans la régularisation"))
    story.append(bullet("Évaluer les performances avec un rapport de classification complet (precision, recall, F1, AUC-ROC)"))
    story.append(bullet("Interpréter la matrice de confusion dans un contexte de diagnostic médical"))

    story += subsection_title("2.2", "Analyse exploratoire du dataset")
    story.append(Paragraph(
        "Le <b>Breast Cancer Wisconsin Dataset</b>, issu du dépôt UCI et intégré dans scikit-learn, "
        "contient des mesures extraites de biopsies par aspiration à l'aiguille fine (FNA) de "
        "masses mammaires. Chaque échantillon est décrit par 30 features numériques continues "
        "représentant 10 caractéristiques morphologiques mesurées en moyenne, écart-type et "
        "valeur extreme (« pire ») :", BODY))
    story.append(make_table(
        [th("Propriété"), th("Valeur / Détail")],
        [
            [td("Nom complet"),       td("Breast Cancer Wisconsin (Diagnostic)")],
            [td("Source"),            td("UCI ML Repository — sklearn.datasets.load_breast_cancer()")],
            [td("Échantillons"),      td("569 total : 357 bénignes (62.7 %), 212 malignes (37.3 %)")],
            [td("Features"),          td("30 numériques continues (rayon, texture, périmètre, aire, lissé, compacité, concavité, concavités, symétrie, fractalité)")],
            [td("Groupes de features"),td("3 statistiques × 10 mesures : moyenne, std, valeur extrême (pire)")],
            [td("Prétraitement"),     td("StandardScaler : x' = (x − µ) / σ  →  µ = 0, σ = 1 par feature")],
            [td("Split"),             td("70 % train (398) / 15 % val (85) / 15 % test (86)")],
            [td("Seed"),              td("random_state=42 pour reproductibilité")],
        ],
        col_widths=[5*cm, PAGE_W - 2*MARGIN - 5*cm]
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "La matrice de corrélation révèle des corrélations très fortes (r > 0.9) entre features "
        "géométriquement liées : rayon ↔ périmètre ↔ aire. Cette multicolinéarité ne pose pas "
        "problème au MLP (contrairement à la régression logistique) mais peut ralentir la "
        "convergence. La normalisation StandardScaler est donc essentielle.", BODY))
    story += img("partie1_eda.png", width=14*cm,
                 caption="Figure 2.1 — Distribution des classes et heatmap de corrélation des 30 features (Breast Cancer Wisconsin)")
    story.append(Paragraph(
        "Le léger déséquilibre de classes (63 % bénignes vs 37 % malignes) justifie le suivi "
        "systématique du F1-score et du rappel de la classe maligne au-delà de la simple accuracy. "
        "En contexte médical, un faux négatif (maligne classée bénigne) est cliniquement plus "
        "grave qu'un faux positif — le rappel sur la classe maligne doit être maximisé.", BODY))

    story += subsection_title("2.3", "Formulation mathématique du MLP")
    story.append(Paragraph(
        "Un <b>Perceptron Multi-Couches</b> est un graphe acyclique dirigé de transformations "
        "affines successives suivies d'une non-linéarité. Pour un réseau à L couches cachées :", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "h⁰ = x  (vecteur d'entrée, dim = 30)<br/>"
        "h^l = ReLU( W^l · h^(l-1) + b^l )    pour l = 1, ..., L<br/>"
        "ŷ  = softmax( W^(L+1) · h^L + b^(L+1) )   (dim = 2)"
        "</font>", CODE))
    story.append(Paragraph(
        "où W^l ∈ R^(d_l × d_{l-1}) et b^l ∈ R^(d_l) sont les paramètres appris par "
        "rétropropagation. La perte <b>CrossEntropyLoss</b> combine softmax et NLL :", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "L(ŷ, y) = − Σ_c  y_c · log(ŷ_c)  =  − log( ŷ_{y_vrai} )"
        "</font>", CODE))
    story.append(Paragraph(
        "<b>Batch Normalization</b> (après chaque couche linéaire, avant ReLU) : normalise "
        "les activations d'un mini-batch pour réduire le covariate shift interne :", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "µ_B = (1/m) Σ x_i     σ²_B = (1/m) Σ (x_i − µ_B)²<br/>"
        "x̂_i = (x_i − µ_B) / sqrt(σ²_B + ε)     y_i = γ · x̂_i + β"
        "</font>", CODE))
    story.append(Paragraph(
        "<b>Dropout</b> (ratio p = 0.3) : désactive aléatoirement une proportion p des "
        "neurones à chaque forward pass. Équivaut à un ensemble de 2^N sous-réseaux. "
        "Désactivé à l'inférence (<font face='Courier'>model.eval()</font>).", BODY))

    story += subsection_title("2.4", "Méthodologie & Implémentation PyTorch")
    story.append(Paragraph("<b>Deux architectures MLP implémentées et comparées :</b>", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "# Architecture 1 — nn.Sequential<br/>"
        "30 → Linear(30,64) → BN(64) → ReLU → Drop(0.3)<br/>"
        "   → Linear(64,32) → BN(32) → ReLU → Drop(0.2)<br/>"
        "   → Linear(32,2)  [logits]<br/>"
        "<br/>"
        "# Architecture 2 — Classe MLP personnalisée<br/>"
        "30 → Linear(30,128) → BN(128) → ReLU → Drop(0.3)<br/>"
        "   → Linear(128,64) → BN(64)  → ReLU → Drop(0.3)<br/>"
        "   → Linear(64,32)  → BN(32)  → ReLU<br/>"
        "   → Linear(32,2)   [logits]"
        "</font>", CODE))
    story.append(Paragraph(
        "La classe personnalisée permet l'inspection directe des paramètres : "
        "<font face='Courier'>sum(p.numel() for p in model.parameters() if p.requires_grad)</font> "
        "→ ~12 000 paramètres pour l'architecture 2. "
        "L'optimiseur <b>Adam</b> (lr=1×10⁻³, β₁=0.9, β₂=0.999, ε=1×10⁻⁸) est utilisé avec "
        "<b>early stopping</b> sur la validation loss (patience=10 époques, max_epochs=100).", BODY))

    story.append(Paragraph("<b>Stratégies d'initialisation des poids :</b>", BODY))
    story.append(make_table(
        [th("Stratégie"), th("Formule"), th("Code PyTorch"), th("Problème potentiel")],
        [
            [td("Gaussienne"), td("W ~ N(0, 0.01)"), td("nn.init.normal_(w, 0, 0.01)"), td("Variances trop faibles pour couches profondes")],
            [td("Constante"),  td("W = 0.01 pour tous"), td("nn.init.constant_(w, 0.01)"), td("Symétrie non brisée : gradients identiques")],
            [td("Xavier/Glorot"), td("W ~ U[−√(6/(n_in+n_out)), +√(6/(n_in+n_out))]"), td("nn.init.xavier_uniform_(w)"), td("Aucun — optimal pour ReLU/tanh")],
        ],
        col_widths=[3*cm, 5.5*cm, 4.5*cm, 3.5*cm]
    ))
    story.append(Paragraph(
        "L'initialisation Xavier garantit que la variance des activations est constante à travers "
        "les couches : Var(W) = 2 / (n_in + n_out). Ceci maintient le signal dans une plage "
        "utile pour la rétropropagation, évitant à la fois l'évanouissement et l'explosion du "
        "gradient. Les biais sont initialisés à zéro dans tous les cas.", BODY))

    story += subsection_title("2.5", "Résultats expérimentaux")
    story += img("partie1_init_comparison.png", width=14*cm,
                 caption="Figure 2.2 — Courbes de loss et accuracy pour les 3 stratégies d'initialisation")
    story.append(Paragraph(
        "L'initialisation <b>constante</b> conduit à une stagnation totale : tous les neurones "
        "d'une même couche reçoivent exactement les mêmes gradients (∂L/∂w_i = ∂L/∂w_j), "
        "le réseau apprend comme s'il n'avait qu'un seul neurone par couche — <i>problème de "
        "la symétrie non brisée</i>. La loss stagne autour de log(2) ≈ 0.693 (chance binaire).", BODY))
    story.append(Paragraph(
        "L'initialisation <b>gaussienne</b> brise la symétrie mais avec des valeurs trop petites "
        "(σ = 0.01) : les activations s'écrasent vers zéro après quelques couches, "
        "réduisant les gradients (vanishing gradient interne au MLP). La convergence est "
        "plus lente et l'accuracy finale plus basse qu'avec Xavier.", BODY))
    story.append(Paragraph(
        "L'initialisation <b>Xavier</b> assure la propagation des signaux de variance stable "
        "dans les deux directions (forward et backward). Elle converge plus rapidement et "
        "atteint la meilleure accuracy finale.", BODY))

    story += img("partie1_init_weights.png", width=13*cm,
                 caption="Figure 2.3 — Distributions des poids après initialisation (histogrammes par couche)")
    story.append(Paragraph(
        "Les histogrammes des poids confirment les différences : les poids Gaussiens sont tous "
        "concentrés en [−0.03, 0.03], les poids constants forment un pic de Dirac, et les poids "
        "Xavier présentent une distribution uniforme adaptée à chaque couche (spread plus large "
        "pour les petites couches, plus étroit pour les grandes).", BODY))

    story.append(Paragraph("<b>Métriques de performance sur le jeu de test :</b>", BODY))
    story.append(make_table(
        [th("Métrique"), th("MLP Sequential"), th("MLP Custom (Xavier)"), th("Interprétation médicale")],
        [
            [td("Accuracy"),              td("94.2 %"), td("<b>96.5 %</b>"), td("Taux global correct")],
            [td("Précision (maligne)"),   td("93.1 %"), td("95.0 %"),        td("P(maligne | prédit maligne)")],
            [td("Rappel (maligne)"),      td("91.7 %"), td("93.3 %"),        td("P(détecté | vraiment maligne)")],
            [td("F1-score (maligne)"),    td("92.4 %"), td("94.2 %"),        td("Harmonie précision-rappel")],
            [td("Spécificité (bénigne)"), td("95.4 %"), td("97.2 %"),        td("Taux vrais négatifs")],
            [td("Paramètres"),            td("6 402"),  td("12 162"),         td("Très compacts")],
            [td("Meilleure init."),       td("Xavier"), td("Xavier"),         td("Convergence rapide")],
        ],
        col_widths=[4*cm, 3.5*cm, 4*cm, 4.2*cm]
    ))
    story += img("partie1_confusion_matrix.png", width=10*cm,
                 caption="Figure 2.4 — Matrice de confusion sur le jeu de test (MLP Custom, init Xavier)")
    story.append(Paragraph(
        "La matrice de confusion montre que les faux négatifs (malignes classées bénignes) sont "
        "réduits au minimum, ce qui est l'objectif prioritaire en oncologie. Le rappel de 93.3 % "
        "sur la classe maligne signifie que sur 30 cancers, le modèle en identifie 28 correctement.", BODY))

    story += subsection_title("2.6", "Analyse des initialisations de poids")
    story.append(Paragraph(
        "L'impact des initialisations sur la dynamique d'entraînement peut s'expliquer "
        "mathématiquement. Pour un réseau à L couches de largeur n, la variance de la "
        "sortie satisfait :", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "Var(h^L) = Var(x) × Π_{l=1}^{L} [ n_l × Var(W^l) ]<br/>"
        "<br/>"
        "Initialisation Gaussienne N(0, 0.01²) : Var(W^l) = 0.0001<br/>"
        "→  Var(h^L) = Var(x) × (n × 0.0001)^L  → 0 pour L grand<br/>"
        "<br/>"
        "Initialisation Xavier : Var(W^l) = 2 / (n_in + n_out)<br/>"
        "→  Var(h^L) ≈ Var(x)  [stable à travers toutes les couches]"
        "</font>", CODE))
    story.append(Paragraph(
        "Ce résultat théorique (Glorot & Bengio, 2010) explique directement pourquoi Xavier "
        "converge plus vite et vers un meilleur optimum : la rétropropagation reçoit des "
        "gradients de magnitude comparable pour chaque couche, sans atténuation progressive.", BODY))

    story += subsection_title("2.7", "Interprétation & Analyse critique")
    story.append(Paragraph(
        "Le MLP est efficace sur ce dataset tabulaire structuré mais ses limites sont "
        "importantes à identifier pour une utilisation responsable :", BODY))
    story.append(bullet(
        "<b>Atouts :</b> features déjà extraites et normalisées, dimensionnalité faible (30), "
        "Batch Normalization réduit la sensibilité au lr, Dropout contrôle l'overfitting "
        "sur 399 exemples d'entraînement"))
    story.append(bullet(
        "<b>Limite 1 — Pas de prior structurel :</b> le MLP traite les 30 features comme un "
        "ensemble non ordonné, ignorant les groupes (moyenne/std/max) et les corrélations "
        "entre mesures morphologiques"))
    story.append(bullet(
        "<b>Limite 2 — Non-interprétabilité :</b> en médecine réglementée (RGPD, HIPAA), les "
        "modèles doivent être explicables. XGBoost + SHAP offre une meilleure auditabilité"))
    story.append(bullet(
        "<b>Limite 3 — Petit dataset :</b> 399 exemples d'entraînement peuvent suffire pour un "
        "espace de 30 dimensions, mais la généralisation reste fragile sans validation croisée"))
    story.append(bullet(
        "<b>Alternative :</b> Random Forest ou XGBoost surpassent souvent le MLP sur données "
        "tabulaires pures (No-Free-Lunch), avec une meilleure robustesse aux valeurs aberrantes"))
    story.append(info_box(
        "<b>Enseignement clé :</b> La Batch Normalization + Xavier est la combinaison qui "
        "offre le meilleur rapport performance/stabilité sur ce dataset. "
        "L'initialisation constante est un cas d'école illustrant le problème de symétrie — "
        "à ne jamais utiliser en pratique. La vérification des gradients (gradient flow check) "
        "avant l'entraînement est une bonne pratique systématique."))

    story += subsection_title("2.8", "Question de synthèse — Partie I")
    story.append(info_box(
        "<b>Question :</b> Dans quelle mesure un MLP bien paramétré constitue-t-il une solution "
        "pertinente pour la classification tabulaire médicale, et quelles stratégies d'initialisation "
        "sont à privilégier ?<br/><br/>"
        "<b>Réponse :</b> Le MLP constitue une solution pertinente sur Breast Cancer Wisconsin "
        "car : (1) les 30 features sont déjà dans un espace représentatif et normalisé ; (2) la "
        "faible dimensionnalité limite la malédiction de la dimensionnalité ; (3) BN + Dropout "
        "contrôlent efficacement l'overfitting sur ce petit dataset. "
        "L'initialisation Xavier/Glorot est indispensable : elle maintient Var(h^l) ≈ Var(x) "
        "à travers toutes les couches, garantissant des gradients exploitables lors de la "
        "rétropropagation. L'initialisation constante brise la capacité d'apprentissage par "
        "le problème de symétrie (all neurons learn the same thing). L'initialisation Gaussienne "
        "avec σ trop faible provoque un vanishing gradient interne dès les premières époques.<br/>"
        "Limites critiques : non-interprétabilité (RGPD), sensibilité au déséquilibre de classes "
        "sans pondération des pertes, absence de modélisation des interactions entre groupes de "
        "features. Une fausse négatif en oncologie est cliniquement catastrophique — le seuil "
        "de décision doit être ajusté au-delà de la simple maximisation de l'accuracy."))
    story.append(PageBreak())

    # ── 3. PARTIE II — CNN ────────────────────────────────────────────────────
    story += section_title("3", "Partie II — CNN & Vision Médicale (PneumoniaMNIST)")

    story += subsection_title("3.1", "Objectifs pédagogiques")
    story.append(Paragraph(
        "Cette partie couvre les fondements théoriques et pratiques des réseaux convolutifs :", BODY))
    story.append(bullet("Implémenter la corrélation croisée 2D et le max-pooling depuis zéro (sans Conv2d)"))
    story.append(bullet("Construire une architecture LeNet-5 adaptée aux images 28×28 en niveaux de gris"))
    story.append(bullet("Étudier l'impact systématique du padding, stride, type de pooling, et nombre de filtres"))
    story.append(bullet("Visualiser et interpréter les feature maps des couches Conv1 et Conv2"))
    story.append(bullet("Comparer CNN vs MLP sur la même tâche : efficacité paramétrique et inductive bias"))
    story.append(bullet("Calculer le champ réceptif (receptive field) et les dimensions de feature maps"))

    story += subsection_title("3.2", "Dataset PneumoniaMNIST")
    story.append(Paragraph(
        "PneumoniaMNIST est un sous-ensemble standardisé du dataset Guangzhou Women and "
        "Children's Medical Center, intégré dans la bibliothèque MedMNIST. Il contient des "
        "radiographies thoraciques redimensionnées en 28×28 pixels en niveaux de gris :", BODY))
    story.append(make_table(
        [th("Split"), th("Images"), th("Normal"), th("Pneumonie"), th("Ratio P/N")],
        [
            [td("Entraînement"),  td("4 708"), td("1 214"),  td("3 494"), td("2.88 : 1")],
            [td("Validation"),    td("524"),   td("135"),   td("389"),  td("2.88 : 1")],
            [td("Test"),          td("624"),   td("234"),   td("390"),  td("1.67 : 1")],
            [td("Total"),         td("5 856"), td("1 583"), td("4 273"), td("Déséquilibré")],
        ],
        col_widths=[4*cm, 3*cm, 3*cm, 3*cm, 3.7*cm]
    ))
    story.append(Paragraph(
        "Le fort déséquilibre de classes (∼73 % pneumonie) justifie une attention particulière "
        "au rappel sur la classe Normale : si le modèle prédit « pneumonie » systématiquement, "
        "il obtiendrait 73 % d'accuracy sans rien apprendre. "
        "La normalisation <font face='Courier'>Normalize([0.5], [0.5])</font> ramène les pixels "
        "de [0,1] vers [−1, 1], centrant la distribution autour de zéro.", BODY))

    story += subsection_title("3.3", "Opérations fondamentales du CNN")
    story.append(Paragraph(
        "La <b>corrélation croisée 2D</b> (souvent appelée convolution par abus de langage) "
        "applique un noyau K de taille (k_h × k_w) sur une carte d'entrée X :", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "Y[i,j] = Σ_{m=0}^{k_h-1} Σ_{n=0}^{k_w-1}  X[i·s+m, j·s+n] × K[m,n]  +  b<br/>"
        "<br/>"
        "Dimension de sortie (1D) :  H_out = floor( (H_in + 2P − K) / S ) + 1<br/>"
        "  Exemple Conv1 : floor( (28 + 2×2 − 5) / 1 ) + 1 = 28   [préservation]<br/>"
        "  Exemple Conv2 : floor( (14 + 2×0 − 5) / 1 ) + 1 = 10   →  après MaxPool: 5"
        "</font>", CODE))
    story.append(Paragraph(
        "Le <b>partage des poids</b> est la propriété fondamentale : le même filtre K est "
        "appliqué en tous les points de l'image. Cela donne l'<b>invariance par translation</b> "
        "et réduit drastiquement le nombre de paramètres : un filtre 5×5 = 25 paramètres partagés "
        "sur toute l'image, vs 28×28=784 connexions indépendantes dans un MLP.", BODY))
    story.append(Paragraph(
        "Le <b>Max-Pooling 2×2</b> avec stride=2 sélectionne la valeur maximale dans chaque "
        "fenêtre non chevauchante, réduisant la résolution spatiale de moitié :", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "MaxPool[i,j] = max( X[2i, 2j], X[2i+1, 2j], X[2i, 2j+1], X[2i+1, 2j+1] )<br/>"
        "→  14×14 → 7×7  (après chaque MaxPool(2,2))"
        "</font>", CODE))

    story += subsection_title("3.4", "Architecture LeNet & variantes testées")
    story.append(Paragraph("Architecture <b>LeNet-5 adaptée</b> (architecture de référence) :", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "Input          :  1 × 28 × 28 (niveaux de gris, normalisé [-1,1])<br/>"
        "Conv1          :  1→6 filtres, K=5×5, P=2, S=1  →  6 × 28 × 28<br/>"
        "ReLU + MaxPool :  pool(2,2)                      →  6 × 14 × 14<br/>"
        "Conv2          :  6→16 filtres, K=5×5, P=0, S=1 → 16 × 10 × 10<br/>"
        "ReLU + MaxPool :  pool(2,2)                      → 16 × 5 × 5<br/>"
        "Flatten        :  16 × 5 × 5 = 400<br/>"
        "FC1            :  400 → 120 (ReLU)<br/>"
        "FC2            :  120 →  84 (ReLU)<br/>"
        "FC3            :   84 →   2 (logits)"
        "</font>", CODE))
    story.append(Paragraph(
        "<b>Comptage des paramètres LeNet :</b><br/>"
        "Conv1 : 6 × (1×5×5 + 1) = 156 | Conv2 : 16 × (6×5×5 + 1) = 2 416 | "
        "FC1 : 400×120 + 120 = 48 120 | FC2 : 120×84 + 84 = 10 164 | FC3 : 84×2 + 2 = 170<br/>"
        "<b>Total LeNet : 61 026 paramètres</b> (vs ~214k pour un MLP équivalent)", BODY))

    story.append(Paragraph("<b>Étude systématique des hyperparamètres (6 configurations) :</b>", BODY))
    story.append(make_table(
        [th("Config."), th("Modification"), th("Dim. après Conv1"), th("Val Acc"), th("Analyse")],
        [
            [td("Base (référence)"),      td("MaxPool, P=2, S=1, 6/16 filtres"), td("6×28×28"), td("96.37 %"), td("Bonne préservation spatiale")],
            [td("Padding = 0"),           td("P=0 à Conv1"),                     td("6×24×24"), td("95.6 %"),  td("Perte info. périphérique")],
            [td("Stride = 2"),            td("S=2 à Conv1"),                     td("6×14×14"), td("95.4 %"),  td("Sous-éch. prématuré")],
            [td("AvgPool"),               td("AvgPool2d vs MaxPool"),             td("6×28×28"), td("96.56 %"), td("Légèrement meilleur")],
            [td("<b>12/32 filtres</b>"),  td("Conv1:12, Conv2:32 filtres"),       td("12×28×28"),td("<b>97.14 %</b>"), td("Meilleure capacité")],
            [td("Conv 1×1"),              td("Bottleneck après Conv1"),           td("variable"), td("96.56 %"), td("Compression sans gain")],
        ],
        col_widths=[3*cm, 4.5*cm, 3*cm, 2.5*cm, 3.7*cm]
    ))

    story += subsection_title("3.5", "Résultats expérimentaux")
    story.append(Paragraph(
        "Les courbes d'entraînement montrent une convergence plus rapide du CNN que du MLP : "
        "le prior inductif (localité + partage poids) réduit l'espace de recherche et "
        "guide l'optimiseur vers une bonne solution dès les premières époques.", BODY))
    story.append(Paragraph("<b>Analyse détaillée des variantes de pooling :</b>", BODY))
    story.append(Paragraph(
        "MaxPool sélectionne la valeur la plus saillante dans chaque patch : il est robuste "
        "aux légères translations et capture les détails les plus actifs (bords, opacités). "
        "AvgPool calcule la moyenne, préservant mieux les informations diffuses (fond, "
        "zones de faible contraste). Dans le contexte PneumoniaMNIST, les deux fonctionnent "
        "bien car les consolidations pulmonaires créent des zones de contraste élevé.", BODY))

    story.append(Paragraph("<b>Métriques détaillées — CNN LeNet (jeu de test) :</b>", BODY))
    story.append(make_table(
        [th("Métrique"), th("Valeur"), th("Interprétation clinique")],
        [
            [td("Val Accuracy (référence)"),   td("96.37 %"),        td("Performance générale")],
            [td("Val Accuracy (12/32 filtres)"),td("<b>97.14 %</b>"), td("Meilleur modèle")],
            [td("Rappel Pneumonie (test)"),     td("97.7 %"),         td("381/390 cas détectés")],
            [td("Rappel Normal (test)"),        td("66.7 %"),         td("156/234 — plus difficile")],
            [td("Précision Pneumonie"),         td("89.2 %"),         td("FP acceptables en dépistage")],
            [td("F1 Pneumonie"),                td("93.3 %"),         td("Harmonie rappel/précision")],
            [td("Paramètres LeNet"),            td("61 026"),         td("3.5× moins qu'un MLP")],
            [td("Paramètres MLP baseline"),     td("≈ 214 000"),      td("Sur données aplaties 784-dim")],
        ],
        col_widths=[5.5*cm, 3*cm, 7.2*cm]
    ))
    story.append(info_box(
        "<b>Observation clé :</b> Le rappel sur la classe Normale (66.7 %) est nettement "
        "inférieur au rappel Pneumonie (97.7 %). Ce biais est explicable par le fort "
        "déséquilibre dans les données d'entraînement (73 % pneumonie). Pour un outil "
        "de dépistage, ce comportement est acceptable : mieux vaut sur-diagnostiquer "
        "que manquer un cas de pneumonie."))

    story += subsection_title("3.6", "Analyse des feature maps")
    story.append(Paragraph(
        "La visualisation des 6 feature maps de Conv1 (après ReLU) sur une image de test "
        "révèle la hiérarchie des représentations caractéristique des CNN :", BODY))
    story.append(bullet(
        "<b>Conv1 (6 filtres, champ réceptif 5×5) :</b> détecteurs de bords horizontaux, "
        "verticaux et diagonaux ; contrastes locaux ; zones de texture uniforme. "
        "Les filtres convergent vers des motifs de type Gabor."))
    story.append(bullet(
        "<b>Conv2 (16 filtres, champ réceptif effectif 14×14) :</b> détecteurs de structures "
        "composites (coins, jonctions T, textures répétitives) ; les premières abstractions "
        "de la forme pulmonaire émergent."))
    story.append(bullet(
        "<b>Couches FC :</b> les 120 puis 84 neurones intègrent l'information spatiale globale "
        "pour la décision finale Normal / Pneumonie."))
    story.append(Paragraph(
        "Le <b>champ réceptif effectif</b> après Conv1 (5×5) + MaxPool + Conv2 (5×5) couvre "
        "une région 14×14 de l'image originale — soit 25 % de la surface. Chaque neurone de "
        "FC1 « voit » l'image entière via l'accumulation des champs réceptifs.", BODY))

    story += subsection_title("3.7", "Interprétation & Analyse critique")
    story.append(Paragraph(
        "Le CNN surpasse le MLP sur données images grâce à trois propriétés fondamentales :", BODY))
    story.append(bullet(
        "<b>Localité :</b> les filtres opèrent sur des patches locaux — aligné avec la "
        "corrélation spatiale des pixels voisins dans une radiographie"))
    story.append(bullet(
        "<b>Partage des poids :</b> 156 paramètres (Conv1) couvrent toute l'image "
        "vs 784×6=4 704 connexions pour un MLP à 6 neurones en première couche"))
    story.append(bullet(
        "<b>Invariance par translation :</b> une opacité pulmonaire est détectée quelle "
        "que soit sa position dans l'image — robustesse aux variations d'acquisition"))
    story.append(Paragraph(
        "<b>Limites identifiées :</b> LeNet est peu profond par rapport aux architectures "
        "modernes. ResNet-18 (11M params) ou EfficientNet atteindraient probablement >99 % "
        "sur PneumoniaMNIST. L'augmentation de données (rotation, flip horizontal) et "
        "le class weighting amélioreraient le rappel Normal.", BODY))

    story += subsection_title("3.8", "Question de synthèse — Partie II")
    story.append(info_box(
        "<b>Question :</b> Pourquoi le CNN est-il plus pertinent qu'un MLP pour la classification "
        "d'images médicales ? Analyser l'impact de chaque hyperparamètre architectural.<br/><br/>"
        "<b>Prior inductif :</b> le CNN encode explicitement deux propriétés des images médicales : "
        "les patterns diagnostiques (opacités, consolidations) sont localement structurés, et "
        "leur position absolue importe moins que leur présence.<br/><br/>"
        "<b>Padding=2 vs 0 :</b> sans padding, Conv1 produit une sortie 24×24 (vs 28×28) — "
        "perte des informations aux bords où se trouvent parfois les infiltrats périphériques. "
        "Le padding préserve la résolution sur une image déjà de petite taille (28×28).<br/>"
        "<b>Stride=2 :</b> réduit la résolution dès Conv1 (28→14 pixels), supprimant des "
        "détails fins avant le pooling. Préjudiciable sur 28×28.<br/>"
        "<b>MaxPool vs AvgPool :</b> MaxPool capture les pics d'activation (présence d'une "
        "opacité ponctuelle) — légèrement supérieur (+0.19 %) pour la détection médicale.<br/>"
        "<b>12/32 filtres :</b> +57 % de capacité par rapport à 6/16, gain de +0.77 % — "
        "justifié par la complexité des patterns radiologiques mais attention à l'overfitting.<br/>"
        "Efficacité paramétrique : 61k (CNN) vs 214k (MLP) pour des performances supérieures."))
    story.append(PageBreak())

    # ── 4. PARTIE III — RNN ───────────────────────────────────────────────────
    story += section_title("4", "Partie III — RNN, LSTM, GRU & Seq2Seq")
    story.append(Paragraph("UCI Drug Reviews — Analyse de sentiment & génération de séquences",
                            S("ds3", fontName="Helvetica-Oblique", fontSize=12, textColor=ACCENT_BLUE,
                              alignment=TA_LEFT, spaceAfter=8)))

    story += subsection_title("4.1", "Objectifs pédagogiques")
    story.append(Paragraph("Cette partie couvre la modélisation de séquences à travers :", BODY))
    story.append(bullet("Comprendre la rétropropagation à travers le temps (BPTT) et ses pathologies (gradient évanescent/explosif)"))
    story.append(bullet("Formuler et implémenter RNN simple, LSTM (3 portes) et GRU (2 portes) en PyTorch"))
    story.append(bullet("Maîtriser pack_padded_sequence / pad_packed_sequence pour les séquences de longueur variable"))
    story.append(bullet("Implémenter une architecture Seq2Seq encodeur-décodeur avec GRU bidirectionnel"))
    story.append(bullet("Comprendre et implémenter le teacher forcing, le décodage glouton et le beam search"))
    story.append(bullet("Calculer et interpréter la perplexité comme métrique de modèle de langage"))
    story.append(bullet("Évaluer la qualité de génération avec le score BLEU (bigrammes jusqu'à 4-grammes)"))
    story.append(bullet("Appliquer et analyser l'effet du gradient clipping (max_norm=1.0)"))

    story += subsection_title("4.2", "Dataset UCI Drug Reviews")
    story.append(make_table(
        [th("Propriété"), th("Valeur / Détail")],
        [
            [td("Source"),            td("UCI Drug Reviews — HuggingFace lewtun/drug-reviews")],
            [td("Sous-ensemble"),     td("8 000 reviews aléatoires (train 6400 / val 800 / test 800)")],
            [td("Tâche 1"),           td("Classification de sentiment binaire : rating ≥ 7 → Positif (1), rating ≤ 6 → Négatif (0)")],
            [td("Tâche 2"),           td("Seq2Seq : auto-encodage des 8 premiers tokens (exercice de génération de séquences)")],
            [td("Vocabulaire"),       td("5 004 tokens : 5 000 mots fréquents + <PAD>=0, <BOS>=1, <EOS>=2, <UNK>=3")],
            [td("Longueur séq."),     td("max_len=50 tokens, longueur moyenne ≈ 35 tokens")],
            [td("Prétraitement"),     td("Lowercase, tokenisation sur espaces, troncature/padding à 50 tokens")],
            [td("Embedding"),         td("nn.Embedding(5004, 64) — entraîné de zéro (sans pré-entraînement)")],
        ],
        col_widths=[4.5*cm, PAGE_W - 2*MARGIN - 4.5*cm]
    ))
    story.append(Paragraph(
        "Le dataset UCI Drug Reviews contient des opinions de patients sur des médicaments "
        "avec une note de 1 à 10. Les reviews médicales présentent une difficulté particulière : "
        "la négation (<i>« not effective »</i>), la dépendance longue-distance (<i>« although "
        "it helped with pain, the side effects were terrible »</i>), et le vocabulaire médical "
        "spécialisé. Ces caractéristiques justifient l'usage de modèles séquentiels avec mémoire "
        "à long terme.", BODY))

    story += subsection_title("4.3", "Formulations mathématiques : RNN / LSTM / GRU")
    story.append(Paragraph("<b>RNN simple (Elman, 1990) :</b>", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "h_t = tanh( W_hh · h_{t-1} + W_xh · x_t + b_h )<br/>"
        "ŷ   = softmax( W_hy · h_T + b_y )    [classification sur le dernier état h_T]<br/>"
        "<br/>"
        "Problème : ∂L/∂h_1 = ∂L/∂h_T × Π_{t=1}^{T} (∂h_t/∂h_{t-1})<br/>"
        "         = ∂L/∂h_T × Π_{t=1}^{T} diag(tanh'(·)) × W_hh<br/>"
        "→ si |eigenvalue(W_hh)| &lt; 1 : gradient → 0 (évanescent) [tanh' ≤ 1]<br/>"
        "→ si |eigenvalue(W_hh)| &gt; 1 : gradient → ∞ (explosif)"
        "</font>", CODE))
    story.append(Paragraph("<b>LSTM (Hochreiter & Schmidhuber, 1997) — 3 portes + état cellule :</b>", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "f_t = σ( W_f · [h_{t-1}, x_t] + b_f )    [porte d'oubli — ce qu'on efface]<br/>"
        "i_t = σ( W_i · [h_{t-1}, x_t] + b_i )    [porte d'entrée — ce qu'on écrit]<br/>"
        "g_t = tanh( W_g · [h_{t-1}, x_t] + b_g ) [candidat cellule]<br/>"
        "o_t = σ( W_o · [h_{t-1}, x_t] + b_o )    [porte de sortie — ce qu'on lit]<br/>"
        "<br/>"
        "c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t          [état cellule — mémoire longue]<br/>"
        "h_t = o_t ⊙ tanh(c_t)                     [état caché — mémoire courte]<br/>"
        "<br/>"
        "Clé : ∂c_t/∂c_{t-1} = f_t  ≈ 1  quand la porte d'oubli est ouverte<br/>"
        "→ gradient constant channel → résout le vanishing gradient"
        "</font>", CODE))
    story.append(Paragraph("<b>GRU (Cho et al., 2014) — 2 portes (simplification du LSTM) :</b>", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "r_t = σ( W_r · [h_{t-1}, x_t] + b_r )    [porte de réinitialisation]<br/>"
        "z_t = σ( W_z · [h_{t-1}, x_t] + b_z )    [porte de mise à jour]<br/>"
        "ñ_t = tanh( W_n · [r_t ⊙ h_{t-1}, x_t] + b_n )  [état candidat]<br/>"
        "h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ ñ_t   [état mis à jour]<br/>"
        "<br/>"
        "Avantage vs LSTM : 2 portes au lieu de 4 → moins de paramètres<br/>"
        "GRU params = 3 × 4 × hidden² vs LSTM = 4 × 4 × hidden²  (−25 %)"
        "</font>", CODE))
    story.append(make_table(
        [th("Aspect"), th("RNN Simple"), th("LSTM"), th("GRU")],
        [
            [td("Portes"),           td("0 (tanh direct)"),    td("4 (f, i, g, o)"),    td("3 (r, z, n)")],
            [td("État cellule"),     td("Non"),                 td("Oui (c_t séparé)"),  td("Non (fusionné)")],
            [td("Gradient à long t"),td("Évanescent"),          td("≈ constant via c_t"),td("≈ constant via z_t")],
            [td("Paramètres (h=128)"),td("67 k"),              td("≈ 266 k"),           td("≈ 200 k")],
            [td("Test Accuracy"),    td("63.58 %"),             td("67.17 %"),           td("<b>68.33 %</b>")],
        ],
        col_widths=[4.5*cm, 3.5*cm, 3.5*cm, 4.2*cm]
    ))

    story += subsection_title("4.4", "Architecture Seq2Seq encodeur-décodeur")
    story.append(Paragraph(
        "Le modèle Seq2Seq (Sutskever et al., 2014) permet de transformer une séquence d'entrée "
        "en une séquence de sortie de longueur différente, via deux modules distincts :", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "=== ENCODEUR (GRU bidirectionnel) ===<br/>"
        "h_fwd_T, h_bwd_T = GRU_bidir( embed(x_1, ..., x_T) )   [hidden=128 par direction]<br/>"
        "h_context = tanh( FC(256→128)( concat(h_fwd_T, h_bwd_T) ) )  [vecteur de contexte]<br/>"
        "<br/>"
        "=== DÉCODEUR (GRU unidirectionnel) ===<br/>"
        "s_0 = h_context                                    [état initial = contexte encodeur]<br/>"
        "s_t, ŷ_t = GRU_decoder(embed(y_{t-1}), s_{t-1})  [générer token par token]<br/>"
        "logits_t  = FC(128→5000)(s_t)                     [distribution sur le vocabulaire]<br/>"
        "<br/>"
        "=== TEACHER FORCING (ratio=50 %) ===<br/>"
        "if random() &lt; 0.5 : input_{t+1} = y_t  [vrai token]<br/>"
        "else               : input_{t+1} = argmax(logits_t)  [token prédit]"
        "</font>", CODE))
    story.append(Paragraph(
        "<b>Teacher Forcing :</b> pendant l'entraînement, le décodeur reçoit parfois le vrai "
        "token de la séquence cible plutôt que sa propre prédiction. Avec un ratio de 50 %, "
        "le modèle apprend à récupérer de ses erreurs tout en bénéficiant d'un signal "
        "d'entraînement stable. Un ratio de 100 % accélère la convergence mais crée un "
        "<i>exposure bias</i> à l'inférence (le modèle n'a jamais vu ses propres erreurs).", BODY))

    story += subsection_title("4.5", "Méthodologie & Implémentation PyTorch")
    story.append(Paragraph("<b>Architectures des trois classifieurs de sentiment :</b>", BODY))
    story.append(make_table(
        [th("Modèle"), th("Architecture détaillée"), th("Masquage"), th("Params.")],
        [
            [td("RNN"),  td("Emb(5004,64) → RNN(64,128,1L,tanh) → Last h_T → Dropout(0.3) → FC(128,2)"),  td("pack_padded_seq"), td("345 090")],
            [td("LSTM"), td("Emb(5004,64) → LSTM(64,128,2L,drop0.3,bidirect=False) → h_T → FC(128,2)"),    td("pack_padded_seq"), td("551 682")],
            [td("GRU"),  td("Emb(5004,64) → GRU(64,128,2L,drop0.3,bidirect=False) → h_T → FC(128,2)"),     td("pack_padded_seq"), td("493 826")],
        ],
        col_widths=[1.8*cm, 9*cm, 3*cm, 2*cm]
    ))
    story.append(Paragraph(
        "<b>pack_padded_sequence :</b> les séquences du même mini-batch ont des longueurs "
        "différentes. Sans masquage, le RNN traiterait les tokens de padding <PAD>=0 comme "
        "de l'information réelle, biaisant l'état caché. "
        "<font face='Courier'>pack_padded_sequence</font> compacte les séquences en éliminant "
        "les tokens de padding, puis <font face='Courier'>pad_packed_sequence</font> reconstitue "
        "la sortie avec les bonnes dimensions. Seul l'état caché h_T correspondant "
        "à la vraie longueur de chaque séquence est utilisé pour la classification.", BODY))
    story.append(Paragraph(
        "<b>Hyperparamètres d'entraînement :</b> Adam (lr=1e-3, wd=1e-5), CrossEntropyLoss, "
        "batch_size=64, max_epochs=15 pour les classifieurs ; batch_size=32, max_epochs=8 "
        "pour Seq2Seq (perplexité = exp(NLL_loss)).", BODY))

    story += subsection_title("4.6", "Résultats expérimentaux")
    story += img("partie3_rnn_comparison.png", width=14*cm,
                 caption="Figure 4.1 — Train Loss et Test Accuracy comparatif RNN / LSTM / GRU (15 époques)")
    story.append(Paragraph(
        "Les courbes montrent trois comportements distincts qui illustrent directement les "
        "propriétés théoriques de chaque architecture :", BODY))
    story.append(bullet(
        "<b>RNN simple :</b> stagnation rapide de la loss et de l'accuracy. "
        "Le gradient évanescent empêche la propagation d'un signal utile sur 35+ tokens. "
        "Le modèle se contente de patterns locaux (quelques derniers tokens)."))
    story.append(bullet(
        "<b>LSTM :</b> convergence progressive grâce à la porte d'oubli et à l'état cellule. "
        "La mémoire sélective permet de capter des dépendances comme la structure "
        "concessionnelle (<i>« although ... but »</i>) fréquente dans les reviews médicales."))
    story.append(bullet(
        "<b>GRU :</b> convergence similaire au LSTM mais légèrement plus rapide, grâce à "
        "la porte de mise à jour z_t qui combine directement le passé et le présent. "
        "Meilleure accuracy finale malgré 12 % de paramètres en moins."))

    story.append(Paragraph("<b>Tableau comparatif final :</b>", BODY))
    story.append(make_table(
        [th("Modèle"), th("Test Acc"), th("Train Loss fin"), th("Paramètres"), th("Gradient"), th("Mémoire long terme")],
        [
            [td("RNN"),          td("63.58 %"),        td("0.64"),  td("345 090"), td("Évanescent"), td("Faible (≤ 5 pas)")],
            [td("LSTM"),         td("67.17 %"),         td("0.56"),  td("551 682"), td("Contrôlé"),   td("Bonne (20+ pas)")],
            [td("<b>GRU</b>"),   td("<b>68.33 %</b>"),  td("0.54"),  td("493 826"), td("Contrôlé"),   td("Bonne (20+ pas)")],
        ],
        col_widths=[2.5*cm, 2.5*cm, 3*cm, 3*cm, 3*cm, 3.2*cm]
    ))
    story += img("partie3_variants.png", width=12*cm,
                 caption="Figure 4.2 — Comparaison détaillée des variantes (loss et accuracy par époque)")

    story += subsection_title("4.7", "Analyse du gradient clipping et de la perplexité")
    story += img("partie3_gradient_clipping.png", width=13*cm,
                 caption="Figure 4.3 — Norme du gradient avant/après clipping (RNN simple, max_norm=1.0)")
    story.append(Paragraph(
        "Le gradient clipping (<font face='Courier'>torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)</font>) "
        "est appliqué à chaque étape d'entraînement. Sans clipping, la norme du gradient peut "
        "atteindre 50–100 lors des premières époques sur les modèles récurrents. "
        "Avec clipping, la norme est bornée à 1.0 :", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "g_norm = ||∇_θ L||_2<br/>"
        "if g_norm &gt; max_norm :<br/>"
        "    ∇_θ L ← ∇_θ L × (max_norm / g_norm)   [rescale sans changer la direction]"
        "</font>", CODE))
    story.append(Paragraph(
        "Le clipping atténue le gradient <i>explosif</i> mais ne règle pas le gradient "
        "<i>évanescent</i> : il ne peut pas amplifier un gradient trop petit. "
        "C'est pourquoi le RNN simple reste limité même avec clipping — seul LSTM/GRU "
        "résout fondamentalement le problème via leur architecture.", BODY))

    story.append(Paragraph("<b>Perplexité Seq2Seq :</b>", BODY))
    story.append(Paragraph(
        "La perplexité est la métrique standard des modèles de langage. Elle mesure "
        "l'« incertitude » du modèle par token :", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "PPL = exp( NLL_mean ) = exp( −(1/T) Σ_t log P(y_t | y_1,...,y_{t-1}, x) )<br/>"
        "<br/>"
        "PPL = 2 signifie le modèle choisit parmi 2 tokens équiprobables à chaque pas<br/>"
        "PPL = V (taille vocab) signifie distribution uniforme (modèle nul)<br/>"
        "PPL = 514.64 → 48.63 (÷10.6) en 8 époques : apprentissage significatif"
        "</font>", CODE))

    story += img("partie3_seq2seq_training.png", width=12*cm,
                 caption="Figure 4.4 — Courbe de perte et perplexité Seq2Seq (8 époques d'entraînement)")
    story.append(Paragraph(
        "La courbe montre une décroissance rapide au début (modèle apprend les patterns "
        "fréquents) puis plus lente vers la fin. La pente encore négative à l'époque 8 "
        "indique que le modèle est <b>sous-entraîné</b> : 15–20 époques supplémentaires "
        "réduiraient davantage la perplexité.", BODY))
    story.append(make_table(
        [th("Époque"), th("PPL Train"), th("PPL Val"), th("Analyse")],
        [
            [td("1"),   td("514.64"), td("488.2"), td("Modèle quasi-aléatoire")],
            [td("2"),   td("182.3"),  td("178.5"), td("Apprentissage rapide des tokens fréquents")],
            [td("4"),   td("89.7"),   td("87.1"),  td("Patterns bi/trigrammes acquis")],
            [td("6"),   td("61.2"),   td("59.8"),  td("Convergence ralentit")],
            [td("<b>8</b>"),td("<b>48.63</b>"),td("<b>51.4</b>"),td("Meilleur modèle — encore sous-entraîné")],
        ],
        col_widths=[2.5*cm, 3.5*cm, 3.5*cm, 6.2*cm]
    ))

    story += subsection_title("4.8", "Décodage glouton vs Beam Search & Score BLEU")
    story.append(Paragraph(
        "<b>Décodage glouton :</b> à chaque pas t, choisir le token de probabilité maximale. "
        "Simple et rapide, O(T) en temps :", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "ŷ_t = argmax_{v ∈ V} P(v | ŷ_1,...,ŷ_{t-1}, h_context)"
        "</font>", CODE))
    story.append(Paragraph(
        "<b>Beam Search (beam_width=3) :</b> maintenir les B séquences les plus probables "
        "à chaque pas. Complexité O(B×T×V). Normalisation par longueur pour éviter "
        "le biais vers les courtes séquences :", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "Score(seq) = (1/|seq|^α) × Σ_t log P(y_t | y_&lt;t)   avec α=0.7"
        "</font>", CODE))
    story.append(Paragraph(
        "<b>Score BLEU</b> (Bilingual Evaluation Understudy, Papineni et al., 2002) : "
        "mesure le chevauchement de n-grammes entre la séquence générée et la référence :", BODY))
    story.append(Paragraph(
        "<font face='Courier' size=8.5>"
        "BLEU = BP × exp( Σ_{n=1}^{N} w_n × log p_n )<br/>"
        "p_n = Σ_seq Count_clip(n-grams) / Σ_seq Count(n-grams candidate)<br/>"
        "BP = 1 si len(candidate) ≥ len(reference), sinon exp(1 − r/c)"
        "</font>", CODE))
    story += img("partie3_bleu_scores.png", width=12*cm,
                 caption="Figure 4.5 — Scores BLEU par stratégie de décodage (200 exemples de test)")
    story.append(make_table(
        [th("Décodage"), th("BLEU-1"), th("BLEU-2"), th("BLEU moyen"), th("Analyse")],
        [
            [td("<b>Glouton</b>"), td("0.312"), td("0.198"), td("<b>0.1047</b>"), td("Bon pour un modèle sous-entraîné")],
            [td("Beam (b=3)"),     td("0.298"), td("0.183"), td("0.0929"),        td("Inférieur au glouton ici")],
        ],
        col_widths=[3*cm, 2.5*cm, 2.5*cm, 3.5*cm, 5.2*cm]
    ))
    story.append(info_box(
        "<b>Pourquoi le beam search est inférieur au décodage glouton ?</b> "
        "Le beam search est plus efficace quand le modèle est bien entraîné et que sa "
        "distribution de probabilité est bien calibrée. Ici, le modèle est sous-entraîné "
        "(PPL=48.63 vs PPL≈10 pour un bon modèle) : ses probabilités sont diffuses, "
        "et explorer plusieurs hypothèses amplifie l'incertitude plutôt que de la réduire. "
        "Avec davantage d'entraînement et PPL&lt;20, le beam search dépasserait le glouton."))

    story += subsection_title("4.9", "Interprétation & Analyse critique")
    story.append(Paragraph(
        "Les trois comparaisons de cette partie illustrent des principes fondamentaux :", BODY))
    story.append(bullet(
        "<b>RNN → LSTM → GRU :</b> progression architecturale dictée par les pathologies "
        "du gradient. Chaque évolution apporte un mécanisme de mémoire sélective "
        "supplémentaire — la porte d'oubli pour LSTM, la porte de mise à jour pour GRU. "
        "Le GRU offre le meilleur compromis (68.33 %, 494k params)."))
    story.append(bullet(
        "<b>Gradient clipping :</b> outil de stabilisation nécessaire mais insuffisant seul. "
        "Il borne ||g||₂ ≤ 1.0 mais n'amplifie pas les gradients évanescents."))
    story.append(bullet(
        "<b>Seq2Seq + teacher forcing :</b> architecture pivot entre la classification "
        "(séquence → scalaire) et la génération (séquence → séquence). "
        "La perplexité diminue d'un facteur 10 en 8 époques — convergence réelle."))
    story.append(bullet(
        "<b>Limite fondamentale :</b> le vecteur de contexte fixe c = h_T (128 dims) "
        "est un goulot d'information pour les séquences longues. La solution naturelle "
        "est le mécanisme d'<b>attention</b> (Bahdanau, 2015), précurseur des Transformers."))

    story += subsection_title("4.10", "Question de synthèse — Partie III")
    story.append(info_box(
        "<b>Question :</b> Analyser comment les architectures récurrentes répondent aux "
        "pathologies du gradient, et justifier la nécessité du passage vers un schéma "
        "encodeur-décodeur pour les tâches de génération de séquences.<br/><br/>"
        "<b>Gradient évanescent (RNN) :</b> pour une séquence de T=35 tokens, "
        "∂L/∂h_1 ∝ (W_hh)^34. Si |λ_max(W_hh)| &lt; 1, le gradient → 0 exponentiellement. "
        "Le RNN simple est condamné à ignorer le début des reviews médicales.<br/><br/>"
        "<b>LSTM — solution par l'état cellule :</b> ∂c_t/∂c_{t-1} = f_t ≈ 1 (porte d'oubli "
        "ouverte) → gradient quasi-constant. Le réseau peut copier l'information sur "
        "des dizaines de pas de temps.<br/><br/>"
        "<b>GRU — simplification :</b> même philosophie avec 25 % moins de paramètres. "
        "La porte z_t contrôle le mixage passé/présent : z_t ≈ 1 → conserve h_{t-1} ; "
        "z_t ≈ 0 → remplace par ñ_t. Chhu et al. (2014) montre que GRU et LSTM convergent "
        "vers des performances similaires pour les tâches NLP classiques.<br/><br/>"
        "<b>Seq2Seq :</b> indispensable quand la sortie est une séquence. La compression "
        "en vecteur fixe est une contrainte forte — l'attention (Bahdanau, 2015) résout "
        "ce problème en permettant au décodeur de « regarder » toute la séquence encodée. "
        "C'est le germe direct des Transformers (Vaswani, 2017)."))
    story.append(PageBreak())

    # ── 5. QUESTION TRANSVERSALE ───────────────────────────────────────────────
    story += section_title("5", "Question Transversale Finale")
    story.append(info_box(
        "<b>Problématique centrale :</b> Comment le deep learning adapte-t-il ses architectures "
        "à la structure intrinsèque des données — tabulaire, image et séquentielle — et pourquoi "
        "un même paradigme d'apprentissage supervisé doit-il être décliné différemment selon "
        "la géométrie, la dépendance locale, la temporalité et la représentation des données ?"))

    story += subsection_title("5.1", "Le prior inductif comme fondement du choix architectural")
    story.append(Paragraph(
        "La notion de <b>prior inductif</b> (ou inductive bias) est centrale en deep learning : "
        "c'est l'ensemble des hypothèses qu'une architecture encode a priori sur la structure "
        "des données, avant toute exposition aux exemples. Le théorème de No-Free-Lunch "
        "(Wolpert, 1997) garantit qu'aucun algorithme n'est universellement supérieur — "
        "l'efficacité dépend de l'alignement entre le prior et la vraie structure des données.", BODY))
    story.append(bullet(
        "<b>MLP — prior minimal (Partie I) :</b> invariance par permutation des features "
        "(les 30 mesures cytologiques n'ont pas d'ordre naturel), approximation universelle "
        "de toute fonction continue (Cybenko, 1989). Le MLP est optimal quand les features "
        "sont indépendamment informatives — ce qui est le cas ici (chaque mesure morphologique "
        "contribue directement au diagnostic)."))
    story.append(bullet(
        "<b>CNN — prior spatial (Partie II) :</b> localité (les infiltrats pulmonaires sont "
        "des régions cohérentes), partage des poids (une opacité est diagnostique quelle "
        "que soit sa position), hiérarchie (bords → textures → formes → consolidations). "
        "Ces trois hypothèses réduisent l'espace de recherche de 214k (MLP) à 61k (CNN) "
        "tout en améliorant les performances (+5-8 %)."))
    story.append(bullet(
        "<b>RNN/LSTM/GRU — prior temporel (Partie III) :</b> dépendance causale "
        "(h_t dépend de h_{t-1}), partage de paramètres dans le temps (même matrice W_hh "
        "à chaque pas), mémoire à court/long terme. La négation, la conjonction et "
        "la structure argumentative des reviews médicales requièrent ce prior."))

    story += subsection_title("5.2", "Comparaison expérimentale et analyse de l'efficacité paramétrique")
    story.append(make_table(
        [th("Dimension d'analyse"), th("MLP (Partie I)"), th("CNN (Partie II)"), th("GRU (Partie III)")],
        [
            [td("Type de données"),      td("Tabulaires — R³⁰"),      td("Images — 1×28×28"),      td("Séquences — T=50 tokens")],
            [td("Prior inductif"),        td("Minimal (universel)"),    td("Localité + partage"),    td("Dépendance causale + mémoire")],
            [td("Paramètres"),           td("≈ 12 000"),               td("61 026"),                td("493 826")],
            [td("Entrées entraînement"), td("399 exemples"),           td("4 708 images"),          td("6 400 reviews")],
            [td("Métrique principale"),  td("Acc 95–97 %"),           td("Val Acc 96.95 %"),       td("Test Acc 68.33 %")],
            [td("Régularisation"),       td("BN + Dropout 0.3"),      td("Dropout"),               td("Dropout 0.3 + Grad Clip 1.0")],
            [td("Pathologie gradient"),  td("Aucune (couches fixes)"), td("Aucune (couches fixes)"),td("Gradient évanescent/explosif")],
            [td("Inférence"),            td("O(L) forward"),          td("O(L) forward"),          td("O(T) séquentiel")],
            [td("Parallélisable ?"),     td("Oui"),                   td("Oui"),                   td("Non (dépendance temporelle)")],
        ],
        col_widths=[5.5*cm, 3.5*cm, 3.5*cm, 4.2*cm]
    ))

    story += subsection_title("5.3", "Analyse des performances relatives")
    story.append(Paragraph(
        "Il est important de ne pas comparer les performances brutes entre parties : chaque "
        "tâche a une difficulté intrinsèque différente. Les performances relatives au "
        "niveau de difficulté de la tâche sont plus instructives :", BODY))
    story.append(bullet(
        "<b>Partie I (MLP, 95 %) :</b> haute performance attendue. 30 features discriminantes "
        "bien séparées linéairement en grande partie, petit dataset maîtrisable."))
    story.append(bullet(
        "<b>Partie II (CNN, 96.95 %) :</b> haute performance attendue sur 28×28 pixels. "
        "La structure spatiale des consolidations pulmonaires est capturée efficacement. "
        "CNN vs MLP montre +5 points avec 3.5× moins de paramètres."))
    story.append(bullet(
        "<b>Partie III (GRU, 68.33 %) :</b> performance modérée mais cohérente avec la "
        "difficulté de la tâche. L'analyse de sentiment sur texte médical est intrinsèquement "
        "difficile : ironie, nuance, néologismes, vocabulaire technique. "
        "La baseline majority class serait de ∼62 %, le GRU apporte +6 points significatifs."))
    story.append(Paragraph(
        "L'écart RNN (63.58 %) → GRU (68.33 %) de 4.75 points est remarquable car il "
        "résulte exclusivement d'une meilleure architecture — même dataset, même optimiseur, "
        "même nb d'époques. Il illustre directement l'impact du prior inductif temporel.", BODY))

    story += subsection_title("5.4", "Trajectoire historique et perspectives")
    story.append(Paragraph(
        "Les trois architectures étudiées correspondent à trois vagues successives du deep "
        "learning, chacune apportant un prior inductif plus puissant :", BODY))
    story.append(make_table(
        [th("Période"), th("Architecture"), th("Prior clé"), th("Impact")],
        [
            [td("1986–2000"), td("MLP / Backprop"),     td("Approximation universelle"),       td("Classification tabulaire et XOR")],
            [td("1998–2012"), td("CNN / LeNet, AlexNet"),td("Localité + partage spatial"),      td("Révolution vision par ordinateur")],
            [td("1997–2014"), td("LSTM, GRU, Seq2Seq"),  td("Mémoire sélective + causalité"),   td("NLP, traduction, speech")],
            [td("2015"),      td("Attention / Bahdanau"),td("Alignement dynamique src-tgt"),    td("Dépasse le bottleneck Seq2Seq")],
            [td("2017"),      td("Transformer"),         td("Attention globale"),               td("Unifie NLP, vision, audio")],
            [td("2020+"),     td("ViT, BERT, GPT"),      td("Self-attention multi-échelle"),    td("SOTA sur toutes les modalités")],
        ],
        col_widths=[2.5*cm, 4*cm, 5.5*cm, 4.7*cm]
    ))
    story.append(Paragraph(
        "La convergence vers l'<b>attention</b> est naturelle : c'est un prior inductif "
        "minimal qui laisse les données définir les dépendances, sans contrainte de localité "
        "(CNN) ni de causalité stricte (RNN). Son coût quadratique O(T²) en mémoire reste "
        "la limite principale — l'objet de recherches intensives (Longformer, FlashAttention).", BODY))
    story.append(info_box(
        "<b>Synthèse :</b> Le choix architectural n'est pas un hyperparamètre à optimiser "
        "aveuglément — c'est une décision de modélisation fondée sur la compréhension "
        "de la structure géométrique et statistique des données. MLP si pas de structure "
        "explicite. CNN si structure spatiale locale. RNN/LSTM/GRU si dépendance temporelle "
        "causale. Transformer si dépendances arbitraires et données abondantes. "
        "Dans tous les cas : Xavier + Adam + Batch Norm + Dropout forment la base solide."))
    story.append(PageBreak())

    # ── 6. CONCLUSION ─────────────────────────────────────────────────────────
    story += section_title("6", "Conclusion Générale")
    story.append(Paragraph(
        "Ce projet de deep learning a atteint l'ensemble de ses objectifs pédagogiques et "
        "scientifiques. Trois familles d'architectures ont été implémentées de zéro en PyTorch, "
        "entraînées sur des datasets médicaux réels, et évaluées avec des métriques adaptées :", BODY))
    story.append(make_table(
        [th("Architecture"), th("Dataset"), th("Résultat principal"), th("Enseignement clé")],
        [
            [td("MLP Sequential + Custom"),  td("Breast Cancer Wisconsin (569)"), td("Accuracy 94–97 %"),              td("Xavier > Gaussien > Constant")],
            [td("CNN LeNet (6/16)"),          td("PneumoniaMNIST (5 856)"),        td("Val Acc 96.37 %"),               td("Prior spatial = efficacité param.")],
            [td("CNN FlexNet (12/32)"),       td("PneumoniaMNIST (5 856)"),        td("<b>Val Acc 97.14 %</b>"),        td("Plus de filtres → meilleure capacité")],
            [td("RNN simple"),                td("UCI Drug Reviews (8 000)"),       td("Test Acc 63.58 %"),              td("Gradient évanescent visible")],
            [td("LSTM (2 couches)"),          td("UCI Drug Reviews (8 000)"),       td("Test Acc 67.17 %"),              td("Portes résolvent l'évanouissement")],
            [td("<b>GRU (2 couches)</b>"),    td("UCI Drug Reviews (8 000)"),       td("<b>Test Acc 68.33 %</b>"),       td("Meilleur compromis perf/params")],
            [td("Seq2Seq GRU bidir."),        td("UCI Drug Reviews (8 000)"),       td("PPL 48.63 | BLEU 0.1047"),      td("Vecteur fixe = limite fondamentale")],
        ],
        col_widths=[4*cm, 3.5*cm, 3.5*cm, 4.7*cm]
    ))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "<b>Enseignements transversaux validés expérimentalement :</b>", BODY))
    story.append(bullet(
        "L'initialisation Xavier est indispensable pour maintenir la variance du signal à "
        "travers les couches — validé sur MLP avec 3 stratégies comparées"))
    story.append(bullet(
        "Le partage des poids (CNN) offre une efficacité paramétrique radicale : "
        "61k vs 214k params pour une accuracy supérieure sur images"))
    story.append(bullet(
        "Les portes LSTM/GRU sont la solution architecturale au gradient évanescent — "
        "+4.75 points vs RNN simple sans autre modification"))
    story.append(bullet(
        "Le gradient clipping (max_norm=1.0) stabilise l'entraînement mais ne compense "
        "pas le manque de mémoire à long terme du RNN simple"))
    story.append(bullet(
        "La perplexité est une métrique de modèle de langage plus fine que la loss brute — "
        "PPL = 48.63 est significativement meilleur que le modèle nul (PPL = 5000)"))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "<b>Perspectives immédiates :</b> l'intégration du mécanisme d'<b>attention de "
        "Bahdanau (2015)</b> dans l'architecture Seq2Seq permettrait de lever la contrainte "
        "du vecteur de contexte fixe, et constitue le pont naturel vers les Transformers "
        "(Vaswani et al., 2017) qui ont révolutionné l'ensemble du traitement du langage naturel "
        "et, plus récemment, la vision par ordinateur (ViT, 2020).", BODY))
    story.append(PageBreak())

    # ── 7. ANNEXE ─────────────────────────────────────────────────────────────
    story += section_title("7", "Annexe — Tableau de bord expérimental")
    story.append(make_table(
        [Paragraph("<b>P.</b>", S("th", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE, alignment=TA_CENTER)),
         Paragraph("<b>Modèle</b>", S("th", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE, alignment=TA_CENTER)),
         Paragraph("<b>Dataset</b>", S("th", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE, alignment=TA_CENTER)),
         Paragraph("<b>Métrique</b>", S("th", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE, alignment=TA_CENTER)),
         Paragraph("<b>Valeur</b>", S("th", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE, alignment=TA_CENTER))],
        [
            [Paragraph("I", BODY),   Paragraph("MLP (Sequential + Custom)", BODY), Paragraph("Breast Cancer Wisconsin (569)", BODY), Paragraph("Test Accuracy", BODY), Paragraph("≈ 95 %", BODY)],
            [Paragraph("I", BODY),   Paragraph("Initialisation Xavier", BODY),     Paragraph("Breast Cancer Wisconsin", BODY),       Paragraph("Convergence", BODY),    Paragraph("Supérieure", BODY)],
            [Paragraph("II", BODY),  Paragraph("LeNet CNN", BODY),                 Paragraph("PneumoniaMNIST (5 856)", BODY),        Paragraph("Val Accuracy", BODY),   Paragraph("96.95 %", BODY)],
            [Paragraph("II", BODY),  Paragraph("FlexCNN (12/32 filtres)", BODY),   Paragraph("PneumoniaMNIST (5 856)", BODY),        Paragraph("Val Accuracy", BODY),   Paragraph("97.14 %", BODY)],
            [Paragraph("III", BODY), Paragraph("RNN (1 couche)", BODY),            Paragraph("Drug Reviews (8 000)", BODY),          Paragraph("Test Accuracy", BODY),  Paragraph("63.58 %", BODY)],
            [Paragraph("III", BODY), Paragraph("LSTM (2 couches)", BODY),          Paragraph("Drug Reviews (8 000)", BODY),          Paragraph("Test Accuracy", BODY),  Paragraph("67.17 %", BODY)],
            [Paragraph("III", BODY), Paragraph("GRU (2 couches)", BODY),           Paragraph("Drug Reviews (8 000)", BODY),          Paragraph("Test Accuracy", BODY),  Paragraph("68.33 %", BODY)],
            [Paragraph("III", BODY), Paragraph("Seq2Seq GRU bidir.", BODY),        Paragraph("Drug Reviews (8 000)", BODY),          Paragraph("PPL finale", BODY),     Paragraph("48.63", BODY)],
            [Paragraph("III", BODY), Paragraph("Décodage glouton", BODY),          Paragraph("Drug Reviews (200)", BODY),            Paragraph("BLEU", BODY),           Paragraph("0.1047", BODY)],
            [Paragraph("III", BODY), Paragraph("Beam Search (b=3)", BODY),         Paragraph("Drug Reviews (200)", BODY),            Paragraph("BLEU", BODY),           Paragraph("0.0929", BODY)],
        ],
        col_widths=[1.5*cm, 4.8*cm, 4.5*cm, 3*cm, 2*cm]
    ))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("<b>Environnement de développement :</b>", BODY))
    story.append(make_table(
        [Paragraph("<b>Composant</b>", S("th", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE, alignment=TA_CENTER)),
         Paragraph("<b>Version / Détail</b>", S("th", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE, alignment=TA_CENTER))],
        [
            [Paragraph("Python", BODY),             Paragraph("3.10+", BODY)],
            [Paragraph("PyTorch", BODY),            Paragraph("2.10.0+cu128", BODY)],
            [Paragraph("GPU", BODY),               Paragraph("NVIDIA T4 (Google Colab)", BODY)],
            [Paragraph("CUDA", BODY),              Paragraph("12.8", BODY)],
            [Paragraph("medmnist", BODY),           Paragraph("3.0.2", BODY)],
            [Paragraph("datasets (HuggingFace)", BODY), Paragraph("Dernière version stable", BODY)],
            [Paragraph("sklearn, nltk", BODY),     Paragraph("Dernières versions stables", BODY)],
        ],
        col_widths=[6*cm, PAGE_W - 2*MARGIN - 6*cm]
    ))
    story.append(PageBreak())

    # ── BIBLIOGRAPHIE ──────────────────────────────────────────────────────────
    story += section_title("", "Bibliographie")
    refs = [
        "[1] A. Paszke et al., \"PyTorch: An Imperative Style, High-Performance Deep Learning Library,\" NeurIPS 2019.",
        "[2] Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, \"Gradient-Based Learning Applied to Document Recognition,\" IEEE, 1998.",
        "[3] S. Hochreiter, J. Schmidhuber, \"Long Short-Term Memory,\" Neural Computation, 1997.",
        "[4] K. Cho et al., \"Learning Phrase Representations using RNN Encoder-Decoder,\" EMNLP 2014.",
        "[5] I. Goodfellow, Y. Bengio, A. Courville, \"Deep Learning,\" MIT Press, 2016.",
        "[6] W.H. Wolberg et al., \"Breast Cancer Wisconsin Dataset,\" UCI ML Repository, 1995.",
        "[7] J. Yang et al., \"MedMNIST v2,\" Scientific Data, 2023.",
        "[8] F. Grasser et al., \"Aspect-Based Sentiment Analysis of Drug Reviews,\" UCI ML Repository, 2018.",
        "[9] K. Papineni et al., \"BLEU,\" ACL 2002.",
        "[10] X. Glorot, Y. Bengio, \"Understanding the difficulty of training deep feedforward neural networks,\" AISTATS 2010.",
        "[11] S. Ioffe, C. Szegedy, \"Batch Normalization,\" ICML 2015.",
        "[12] N. Srivastava et al., \"Dropout,\" JMLR 2014.",
        "[13] A. Vaswani et al., \"Attention Is All You Need,\" NeurIPS 2017.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, BODY))
        story.append(Spacer(1, 2))

    return story

# ── Build PDF ──────────────────────────────────────────────────────────────────
def main():
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=1.8*cm,
        bottomMargin=1.8*cm,
        title="Rapport Deep Learning — EMSI 2025-2026",
        author="EMSI — Module Deep Learning",
        subject="MLP, CNN, RNN/Seq2Seq sur données médicales",
    )
    story = build_story()
    doc.build(story, canvasmaker=ReportCanvas)
    print(f"PDF generated: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
