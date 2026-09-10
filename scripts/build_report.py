"""Build the engineering PDF from Markdown, benchmark data and pytest XML."""

import html
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import os

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib")
)
import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    Image,
    Preformatted,
)

ROOT = Path(__file__).resolve().parents[1]
FONT = Path(matplotlib.get_data_path()) / "fonts/ttf"
for name, file in [
    ("Body", "DejaVuSans.ttf"),
    ("BodyBold", "DejaVuSans-Bold.ttf"),
    ("Mono", "DejaVuSansMono.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(FONT / file)))
pdfmetrics.registerFontFamily(
    "Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold"
)
BLUE = colors.HexColor("#005bff")
DARK = colors.HexColor("#182b49")
GRAY = colors.HexColor("#62718a")
styles = {
    "body": ParagraphStyle(
        "body",
        fontName="Body",
        fontSize=9.2,
        leading=13.3,
        spaceAfter=7,
        textColor=DARK,
    ),
    "h1": ParagraphStyle(
        "h1",
        fontName="BodyBold",
        fontSize=19,
        leading=24,
        spaceAfter=15,
        textColor=DARK,
        keepWithNext=True,
    ),
    "h2": ParagraphStyle(
        "h2",
        fontName="BodyBold",
        fontSize=14,
        leading=19,
        spaceAfter=12,
        textColor=BLUE,
        keepWithNext=True,
    ),
    "h3": ParagraphStyle(
        "h3",
        fontName="BodyBold",
        fontSize=11,
        leading=15,
        spaceBefore=7,
        spaceAfter=7,
        textColor=DARK,
        keepWithNext=True,
    ),
    "table": ParagraphStyle(
        "table", fontName="Body", fontSize=8.1, leading=11, textColor=DARK
    ),
    "th": ParagraphStyle(
        "th", fontName="BodyBold", fontSize=8.2, leading=11, textColor=colors.white
    ),
    "caption": ParagraphStyle(
        "caption", fontName="Body", fontSize=8, leading=11, textColor=GRAY, spaceAfter=9
    ),
    "code": ParagraphStyle(
        "code",
        fontName="Mono",
        fontSize=7.6,
        leading=10.4,
        textColor=DARK,
        backColor=colors.HexColor("#f2f5fa"),
        borderPadding=8,
        spaceAfter=12,
    ),
}


def inline(s):
    # Turn links into named clickable references; no raw URL wrapping in body.
    links = []

    def link(m):
        links.append((m.group(1), m.group(2)))
        return f"LINKTOKEN{len(links) - 1}END"

    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", link, s)
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r'<font name="Mono">\1</font>', s)
    for i, (label, url) in enumerate(links):
        s = s.replace(
            f"LINKTOKEN{i}END",
            f'<link href="{html.escape(url, quote=True)}" color="#005bff">{html.escape(label)}</link>',
        )
    return s


def footer(c, doc):
    w, h = A4
    c.setStrokeColor(colors.HexColor("#dce4f0"))
    c.line(42, 40, w - 42, 40)
    c.setFont("Body", 7.4)
    c.setFillColor(GRAY)
    c.drawString(42, 27, "Ozon Tech - Иннополис · Измерение габаритов · Kr1ny")
    c.drawRightString(w - 42, 27, str(doc.page))
    if doc.page > 1:
        c.setFont("Body", 7.2)
        c.drawString(42, h - 24, "КОМПЬЮТЕРНОЕ ЗРЕНИЕ / ВАРИАНТ 1")


def build():
    summary = json.loads((ROOT / "results/summary.json").read_text())
    tests = ET.parse(ROOT / "results/tests.xml").getroot()
    suites = tests.findall("testsuite") if tests.tag == "testsuites" else [tests]
    count = sum(int(s.get("tests", "0")) for s in suites)
    failures = sum(
        int(s.get("failures", "0")) + int(s.get("errors", "0")) for s in suites
    )
    if failures:
        raise RuntimeError("Report requires passing tests")
    geometry = json.loads((ROOT / "results/geometry_crosscheck.json").read_text())
    load = json.loads((ROOT / "results/load_benchmark.json").read_text())
    if not geometry["passed"] or not all(row["passed"] for row in load["runs"]):
        raise RuntimeError("Report requires passing geometry and load checks")
    large = next(row for row in load["runs"] if row["points"] == 3_700_000)
    curved = next(
        row
        for row in load["runs"]
        if row["shape"] == "ellipsoid_surface" and row["points"] == 3000
    )
    values = {
        **summary,
        "tests_count": count,
        "platform": summary["environment"]["platform"],
        "python_version": summary["environment"]["python"],
        "load_max_ms": large["obb_ms"],
        "curved_median_ms": curved["obb_ms"],
        "curved_max_ms": curved["max_ms"],
        "geometry_cases": geometry["cases"],
        "geometry_max_ratio": f"{geometry['max_ratio_to_best_de']:.6f}".replace(
            ".", ","
        ),
    }
    raw = (ROOT / "docs/report.md").read_text()
    for key, value in values.items():
        if isinstance(value, float):
            value = f"{value:.2f}".replace(".", ",")
        raw = raw.replace("{{" + key + "}}", str(value))
    if "{{" in raw:
        raise RuntimeError("Unresolved report placeholders")
    raw = raw.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    (ROOT / "results/report_resolved.md").write_text(raw)
    readme = ROOT / "README.md"
    results_text = (
        f"В комплекте {summary['total_cases']} синтетических сценариев. "
        f"В штатной группе из {summary['reference_evaluable_cases']} проходов все диагностические кандидаты "
        f"уложились в допуск `max(5%, 5 мм)`; максимальная ошибка {summary['error_max_mm']:.2f} мм. "
        f"Контроль качества принял {summary['nominal_accepted_count']} прохода "
        f"({100 * summary['nominal_acceptance_rate']:.1f}%), ещё {summary['nominal_review_count']} получили `review`. "
        f"Во всей серии: {summary['review_count']} результатов `review`, "
        f"ошибочных принятий среди {summary['accepted_reference_cases']} принятых случаев с эталоном "
        f"- {summary['accepted_reference_failure_count']}. Полная статистика: [benchmark.csv](results/benchmark.csv).\n\n"
        f"Успешно выполнены {count} автоматических тестов. Независимый численный поиск проверяет "
        f"{geometry['cases']} полных облаков в той же конфигурации, что использует CLI. "
        f"Максимальное превышение объёма относительно лучшего независимого решения "
        f"- {100 * (geometry['max_ratio_to_best_de'] - 1):.3f}%. "
        "Точки и найденные коробки: [geometry_crosscheck.json](results/geometry_crosscheck.json).\n\n"
        "Скрытый сбой энкодера, одиночные и групповые ложные отражения, недостаточные ракурсы "
        "и отсутствующие подтверждения качества направляются на повторную проверку. "
        "Результаты относятся к синтетической модели и указанным контрольным фигурам."
    )
    updated, n = re.subn(
        r"<!-- results:start -->.*?<!-- results:end -->",
        "<!-- results:start -->\n" + results_text + "\n<!-- results:end -->",
        readme.read_text(),
        flags=re.S,
    )
    if n != 1:
        raise RuntimeError("Expected one generated results block in README")
    readme.write_text(updated)
    lines = raw.splitlines()
    story = []
    i = 0
    width = A4[0] - 84
    section = ""
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            story.append(
                Preformatted("\n".join(block), styles["code"], maxLineLength=91)
            )
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                parts = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", x) for x in parts):
                    rows.append(parts)
                i += 1
            columns = len(rows[0])
            weights = {
                2: [0.37, 0.63],
                3: [0.24, 0.38, 0.38],
                4: [0.19, 0.29, 0.22, 0.30],
            }[columns]
            if section.startswith("10.") and columns == 2:
                weights = [0.65, 0.35]
            if section.startswith("5.") and columns == 3:
                weights = [0.30, 0.28, 0.42]
            data = [
                [
                    Paragraph(inline(c), styles["th"] if r == 0 else styles["table"])
                    for c in row
                ]
                for r, row in enumerate(rows)
            ]
            table = Table(
                data,
                colWidths=[width * x for x in weights],
                repeatRows=1,
                hAlign="LEFT",
            )
            padding = 5 if section.startswith("13.") else 6
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.HexColor("#f0f4fa"), colors.white],
                        ),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), padding),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
                        (
                            "LINEBELOW",
                            (0, -1),
                            (-1, -1),
                            0.5,
                            colors.HexColor("#dce4f0"),
                        ),
                    ]
                )
            )
            story.extend([table, Spacer(1, 10)])
            continue
        img = re.fullmatch(r"!\[([^\]]+)\]\(([^)]+)\)", line)
        if img:
            path = ROOT / "docs" / img.group(2)
            from PIL import Image as PILImage

            with PILImage.open(path) as im:
                w, h = im.size
            image_width = width * 0.9 if "station_top" in str(path) else width
            story.extend(
                [
                    Image(str(path), width=image_width, height=image_width * h / w),
                    Spacer(1, 7),
                ]
            )
            i += 1
            continue
        heading = re.match(r"^(#{1,3}) (.+)", line)
        if heading:
            if len(heading.group(1)) == 1:
                section = heading.group(2)
            story.append(
                Paragraph(
                    inline(heading.group(2)), styles["h" + str(len(heading.group(1)))]
                )
            )
            i += 1
            continue
        paragraph = [line]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].startswith(("#", "|", "```", "<!--", "!["))
        ):
            paragraph.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline(" ".join(paragraph)), styles["body"]))
    output = ROOT / "output/pdf/ozon_dimensioning_report.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=43,
        bottomMargin=54,
        title="Измерение габаритов товаров на конвейере",
        author="Kr1ny",
        subject="Ozon Tech - Иннополис, вариант 1",
        pageCompression=1,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == "__main__":
    build()
