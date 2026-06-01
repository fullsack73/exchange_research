#!/usr/bin/env python3
"""
Build the DAT capstone DOCX from the Markdown draft using LibreOffice UNO.

Recommended:
    /Applications/LibreOffice.app/Contents/Resources/python \
        reports/build_capstone_docx.py \
        reports/capstone_report_draft.md \
        reports/DAT_캡스톤_프로젝트_보고서_양식.docx \
        reports/capstone_report_generated.docx

This machine may not expose pyuno to system Python. In that case the default
`--backend auto` falls back to direct OOXML generation while preserving the
template package, styles, headers, footers, and page settings.

Fallback on macOS system Python, if pyuno can be imported:
    python3 reports/build_capstone_docx.py ...
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


EMU_PER_CM = 360000
HMM_PER_CM = 1000  # LibreOffice sizes are 1/100 mm.
PARAGRAPH_BREAK = 0

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
PR_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"

for prefix, uri in {
    "w": W_NS,
    "r": R_NS,
    "a": A_NS,
    "wp": WP_NS,
    "pic": PIC_NS,
}.items():
    ET.register_namespace(prefix, uri)


@dataclass
class Block:
    kind: str
    text: str = ""
    level: int = 0
    rows: list[list[str]] | None = None
    image_path: str = ""
    alt: str = ""


def parse_markdown(path: Path) -> list[Block]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[Block] = []
    paragraph: list[str] = []
    i = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append(Block("paragraph", " ".join(paragraph).strip()))
            paragraph = []

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if stripped == "---":
            flush_paragraph()
            blocks.append(Block("page_break"))
            i += 1
            continue

        image_match = re.match(r"^!\[(.*?)\]\((.*?)\)$", stripped)
        if image_match:
            flush_paragraph()
            blocks.append(
                Block(
                    "image",
                    image_path=image_match.group(2),
                    alt=image_match.group(1),
                )
            )
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            rows: list[list[str]] = []
            while i < len(lines):
                row_line = lines[i].strip()
                if not (row_line.startswith("|") and row_line.endswith("|")):
                    break
                cells = [cell.strip() for cell in row_line.strip("|").split("|")]
                # Drop Markdown separator row.
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    rows.append(cells)
                i += 1
            if rows:
                blocks.append(Block("table", rows=rows))
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            blocks.append(
                Block(
                    "heading",
                    text=heading_match.group(2).strip(),
                    level=len(heading_match.group(1)),
                )
            )
            i += 1
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            blocks.append(Block("bullet", stripped[2:].strip()))
            i += 1
            continue

        paragraph.append(stripped)
        i += 1

    flush_paragraph()
    return blocks


def png_size(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            sig = f.read(24)
        if sig[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        width, height = struct.unpack(">II", sig[16:24])
        return width, height
    except OSError:
        return None


def qn(namespace: str, tag: str) -> str:
    return f"{{{namespace}}}{tag}"


def w(tag: str) -> str:
    return qn(W_NS, tag)


def r(tag: str) -> str:
    return qn(R_NS, tag)


def attr(namespace: str, name: str) -> str:
    return qn(namespace, name)


def text_el(text: str) -> ET.Element:
    t = ET.Element(w("t"))
    if text.startswith(" ") or text.endswith(" "):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return t


def paragraph_xml(text: str = "", style: str | None = None, bold: bool = False) -> ET.Element:
    p = ET.Element(w("p"))
    if style:
        ppr = ET.SubElement(p, w("pPr"))
        pstyle = ET.SubElement(ppr, w("pStyle"))
        pstyle.set(w("val"), style)
    run = ET.SubElement(p, w("r"))
    if bold:
        rpr = ET.SubElement(run, w("rPr"))
        ET.SubElement(rpr, w("b"))
    run.append(text_el(text))
    return p


def page_break_xml() -> ET.Element:
    p = ET.Element(w("p"))
    r_el = ET.SubElement(p, w("r"))
    br = ET.SubElement(r_el, w("br"))
    br.set(w("type"), "page")
    return p


def table_xml(rows: list[list[str]]) -> ET.Element:
    tbl = ET.Element(w("tbl"))
    tbl_pr = ET.SubElement(tbl, w("tblPr"))
    tbl_w = ET.SubElement(tbl_pr, w("tblW"))
    tbl_w.set(w("w"), "0")
    tbl_w.set(w("type"), "auto")
    borders = ET.SubElement(tbl_pr, w("tblBorders"))
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = ET.SubElement(borders, w(side))
        border.set(w("val"), "single")
        border.set(w("sz"), "4")
        border.set(w("space"), "0")
        border.set(w("color"), "808080")
    for row_idx, row in enumerate(rows):
        tr = ET.SubElement(tbl, w("tr"))
        for cell_text in row:
            tc = ET.SubElement(tr, w("tc"))
            tc_pr = ET.SubElement(tc, w("tcPr"))
            if row_idx == 0:
                shd = ET.SubElement(tc_pr, w("shd"))
                shd.set(w("fill"), "F2F2F2")
            tc.append(paragraph_xml(cell_text, bold=(row_idx == 0)))
    return tbl


def image_xml(rid: str, alt: str, image_path: Path, docpr_id: int) -> ET.Element:
    size = png_size(image_path) or (1000, 600)
    width_px, height_px = size
    max_width_emu = int(15 * EMU_PER_CM)
    max_height_emu = int(9 * EMU_PER_CM)
    # Assume 96 DPI when image metadata is unavailable.
    width_emu = int(width_px / 96 * 914400)
    height_emu = int(height_px / 96 * 914400)
    scale = min(max_width_emu / width_emu, max_height_emu / height_emu, 1.0)
    cx = int(width_emu * scale)
    cy = int(height_emu * scale)

    p = ET.Element(w("p"))
    ppr = ET.SubElement(p, w("pPr"))
    jc = ET.SubElement(ppr, w("jc"))
    jc.set(w("val"), "center")
    run = ET.SubElement(p, w("r"))
    drawing = ET.SubElement(run, w("drawing"))
    inline = ET.SubElement(drawing, qn(WP_NS, "inline"))
    extent = ET.SubElement(inline, qn(WP_NS, "extent"))
    extent.set("cx", str(cx))
    extent.set("cy", str(cy))
    effect_extent = ET.SubElement(inline, qn(WP_NS, "effectExtent"))
    for key in ("l", "t", "r", "b"):
        effect_extent.set(key, "0")
    doc_pr = ET.SubElement(inline, qn(WP_NS, "docPr"))
    doc_pr.set("id", str(docpr_id))
    doc_pr.set("name", alt or f"Picture {docpr_id}")
    c_nv = ET.SubElement(inline, qn(WP_NS, "cNvGraphicFramePr"))
    locks = ET.SubElement(c_nv, qn(A_NS, "graphicFrameLocks"))
    locks.set("noChangeAspect", "1")
    graphic = ET.SubElement(inline, qn(A_NS, "graphic"))
    graphic_data = ET.SubElement(graphic, qn(A_NS, "graphicData"))
    graphic_data.set("uri", PIC_NS)
    pic = ET.SubElement(graphic_data, qn(PIC_NS, "pic"))
    nv_pic_pr = ET.SubElement(pic, qn(PIC_NS, "nvPicPr"))
    c_nv_pr = ET.SubElement(nv_pic_pr, qn(PIC_NS, "cNvPr"))
    c_nv_pr.set("id", "0")
    c_nv_pr.set("name", image_path.name)
    ET.SubElement(nv_pic_pr, qn(PIC_NS, "cNvPicPr"))
    blip_fill = ET.SubElement(pic, qn(PIC_NS, "blipFill"))
    blip = ET.SubElement(blip_fill, qn(A_NS, "blip"))
    blip.set(r("embed"), rid)
    stretch = ET.SubElement(blip_fill, qn(A_NS, "stretch"))
    ET.SubElement(stretch, qn(A_NS, "fillRect"))
    sp_pr = ET.SubElement(pic, qn(PIC_NS, "spPr"))
    xfrm = ET.SubElement(sp_pr, qn(A_NS, "xfrm"))
    off = ET.SubElement(xfrm, qn(A_NS, "off"))
    off.set("x", "0")
    off.set("y", "0")
    ext = ET.SubElement(xfrm, qn(A_NS, "ext"))
    ext.set("cx", str(cx))
    ext.set("cy", str(cy))
    prst = ET.SubElement(sp_pr, qn(A_NS, "prstGeom"))
    prst.set("prst", "rect")
    ET.SubElement(prst, qn(A_NS, "avLst"))
    return p


def resolve_image(markdown_path: Path, image_path: str) -> Path:
    candidate = Path(image_path)
    if not candidate.is_absolute():
        candidate = markdown_path.parent / candidate
    return candidate.resolve()


def import_uno():
    try:
        import uno  # type: ignore
        from com.sun.star.beans import PropertyValue  # type: ignore
        from com.sun.star.awt import Size  # type: ignore
        from com.sun.star.text.TextContentAnchorType import (  # type: ignore
            AS_CHARACTER,
        )
        from com.sun.star.style.ParagraphAdjust import CENTER  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on LibreOffice runtime.
        raise RuntimeError(
            "LibreOffice UNO is not available. Run this script with "
            "`/Applications/LibreOffice.app/Contents/Resources/python`, or set "
            "PYTHONPATH/URE_BOOTSTRAP so `import uno` works."
        ) from exc

    return uno, PropertyValue, Size, AS_CHARACTER, CENTER


def prop(name: str, value, PropertyValue):
    p = PropertyValue()
    p.Name = name
    p.Value = value
    return p


def set_style(cursor, style_name: str) -> None:
    try:
        cursor.ParaStyleName = style_name
    except Exception:
        pass


def insert_paragraph(text, cursor, content: str, style: str | None = None) -> None:
    if style:
        set_style(cursor, style)
    text.insertString(cursor, content, False)
    text.insertControlCharacter(cursor, PARAGRAPH_BREAK, False)


def insert_heading(text, cursor, content: str, level: int) -> None:
    style = {1: "Heading 1", 2: "Heading 2", 3: "Heading 3"}.get(level, "Heading 4")
    insert_paragraph(text, cursor, content, style)


def insert_bullet(text, cursor, content: str) -> None:
    insert_paragraph(text, cursor, f"• {content}")


def insert_table(doc, text, cursor, rows: list[list[str]]) -> None:
    if not rows:
        return
    columns = max(len(row) for row in rows)
    normalized = [row + [""] * (columns - len(row)) for row in rows]
    table = doc.createInstance("com.sun.star.text.TextTable")
    table.initialize(len(normalized), columns)
    text.insertTextContent(cursor, table, False)

    names = table.getCellNames()
    for row_idx, row in enumerate(normalized):
        for col_idx, cell_text in enumerate(row):
            cell = table.getCellByName(names[row_idx * columns + col_idx])
            cell.setString(cell_text)
            try:
                cell.VertOrient = 1
            except Exception:
                pass

    # Light formatting: repeat first row as header where supported.
    try:
        table.RepeatHeadline = True
        table.HeaderRowCount = 1
    except Exception:
        pass

    text.insertControlCharacter(cursor, PARAGRAPH_BREAK, False)


def insert_image(doc, text, cursor, markdown_path: Path, image_path: str, alt: str) -> None:
    uno, _, Size, AS_CHARACTER, CENTER = import_uno()
    resolved = resolve_image(markdown_path, image_path)
    if not resolved.exists():
        insert_paragraph(text, cursor, f"[이미지 누락: {image_path}]")
        return

    graphic = doc.createInstance("com.sun.star.text.TextGraphicObject")
    graphic.GraphicURL = uno.systemPathToFileUrl(str(resolved))
    graphic.AnchorType = AS_CHARACTER

    pixels = png_size(resolved)
    if pixels:
        width_px, height_px = pixels
        max_width = 15 * HMM_PER_CM
        max_height = 9 * HMM_PER_CM
        scale = min(max_width / width_px, max_height / height_px, 1.0)
        graphic.Size = Size(int(width_px * scale), int(height_px * scale))

    try:
        cursor.ParaAdjust = CENTER
    except Exception:
        pass
    text.insertTextContent(cursor, graphic, False)
    text.insertControlCharacter(cursor, PARAGRAPH_BREAK, False)
    if alt:
        insert_paragraph(text, cursor, alt)


def clear_document_body(doc) -> None:
    doc.Text.setString("")


def update_indexes(doc) -> None:
    try:
        for index in doc.getDocumentIndexes():
            index.update()
    except Exception:
        pass


def build_docx(markdown_path: Path, template_path: Path, output_path: Path) -> None:
    uno, PropertyValue, _, _, _ = import_uno()

    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_ctx
    )
    soffice_proc = None
    try:
        ctx = connect_or_start_soffice(resolver)
    except Exception:
        soffice_proc = start_soffice()
        ctx = connect_or_start_soffice(resolver)
    smgr = ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)

    template_url = uno.systemPathToFileUrl(str(template_path.resolve()))
    output_url = uno.systemPathToFileUrl(str(output_path.resolve()))
    load_props = (
        prop("Hidden", True, PropertyValue),
        prop("ReadOnly", False, PropertyValue),
    )
    doc = desktop.loadComponentFromURL(template_url, "_blank", 0, load_props)
    if doc is None:
        raise RuntimeError(f"Failed to open template: {template_path}")

    try:
        clear_document_body(doc)
        text = doc.Text
        cursor = text.createTextCursor()

        for block in parse_markdown(markdown_path):
            if block.kind == "heading":
                insert_heading(text, cursor, block.text, block.level)
            elif block.kind == "paragraph":
                insert_paragraph(text, cursor, block.text)
            elif block.kind == "bullet":
                insert_bullet(text, cursor, block.text)
            elif block.kind == "table":
                insert_table(doc, text, cursor, block.rows or [])
            elif block.kind == "image":
                insert_image(doc, text, cursor, markdown_path, block.image_path, block.alt)
            elif block.kind == "page_break":
                try:
                    cursor.BreakType = 4  # com.sun.star.style.BreakType.PAGE_BEFORE
                except Exception:
                    pass

        update_indexes(doc)
        store_props = (
            prop(
                "FilterName",
                "Office Open XML Text",
                PropertyValue,
            ),
        )
        doc.storeAsURL(output_url, store_props)
    finally:
        doc.close(True)
        if soffice_proc is not None:
            soffice_proc.terminate()


def find_soffice() -> str:
    candidates = [
        os.environ.get("SOFFICE"),
        "/usr/local/bin/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return "soffice"


def start_soffice() -> subprocess.Popen:
    cmd = [
        find_soffice(),
        "--headless",
        "--nologo",
        "--nodefault",
        "--nofirststartwizard",
        "--accept=socket,host=localhost,port=2002;urp;StarOffice.ComponentContext",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def connect_or_start_soffice(resolver):
    last_error = None
    for _ in range(30):
        try:
            return resolver.resolve(
                "uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
            )
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError("Could not connect to LibreOffice UNO server") from last_error


def next_relationship_id(root: ET.Element) -> int:
    max_id = 0
    for rel in root:
        rid = rel.attrib.get("Id", "")
        if rid.startswith("rId") and rid[3:].isdigit():
            max_id = max(max_id, int(rid[3:]))
    return max_id + 1


def ensure_png_content_type(root: ET.Element) -> None:
    for child in root:
        if child.tag == qn(CT_NS, "Default") and child.attrib.get("Extension") == "png":
            return
    default = ET.Element(qn(CT_NS, "Default"))
    default.set("Extension", "png")
    default.set("ContentType", "image/png")
    root.insert(0, default)


def build_docx_ooxml(markdown_path: Path, template_path: Path, output_path: Path) -> None:
    blocks = parse_markdown(markdown_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(template_path, "r") as zin:
        document_root = ET.fromstring(zin.read("word/document.xml"))
        rels_root = ET.fromstring(zin.read("word/_rels/document.xml.rels"))
        content_types_root = ET.fromstring(zin.read("[Content_Types].xml"))
        existing_names = set(zin.namelist())
        for key in list(document_root.attrib):
            if key.startswith(f"{{{MC_NS}}}"):
                del document_root.attrib[key]

        body = document_root.find(w("body"))
        if body is None:
            raise RuntimeError("Template document.xml has no body")
        sect_pr = body.find(w("sectPr"))
        for child in list(body):
            body.remove(child)

        next_rid = next_relationship_id(rels_root)
        next_docpr = 1
        media_to_add: list[tuple[str, Path]] = []

        for block in blocks:
            if block.kind == "heading":
                style = {1: "Heading1", 2: "Heading2", 3: "Heading3"}.get(
                    block.level, "Heading4"
                )
                body.append(paragraph_xml(block.text, style=style))
            elif block.kind == "paragraph":
                body.append(paragraph_xml(block.text))
            elif block.kind == "bullet":
                body.append(paragraph_xml(f"• {block.text}"))
            elif block.kind == "page_break":
                body.append(page_break_xml())
            elif block.kind == "table":
                body.append(table_xml(block.rows or []))
            elif block.kind == "image":
                image_path = resolve_image(markdown_path, block.image_path)
                if image_path.exists():
                    rid = f"rId{next_rid}"
                    next_rid += 1
                    media_name = f"word/media/capstone_image_{next_docpr}.png"
                    rel = ET.SubElement(rels_root, qn(PR_NS, "Relationship"))
                    rel.set("Id", rid)
                    rel.set(
                        "Type",
                        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                    )
                    rel.set("Target", f"media/capstone_image_{next_docpr}.png")
                    media_to_add.append((media_name, image_path))
                    body.append(image_xml(rid, block.alt, image_path, next_docpr))
                    if block.alt:
                        body.append(paragraph_xml(block.alt))
                    next_docpr += 1
                else:
                    body.append(paragraph_xml(f"[이미지 누락: {block.image_path}]"))

        if sect_pr is not None:
            body.append(sect_pr)
        ensure_png_content_type(content_types_root)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                name = item.filename
                if name in {
                    "word/document.xml",
                    "word/_rels/document.xml.rels",
                    "[Content_Types].xml",
                }:
                    continue
                if name in {media_name for media_name, _ in media_to_add}:
                    continue
                zout.writestr(item, zin.read(name))
            zout.writestr(
                "word/document.xml",
                ET.tostring(document_root, encoding="utf-8", xml_declaration=True),
            )
            ET.register_namespace("", PR_NS)
            zout.writestr(
                "word/_rels/document.xml.rels",
                ET.tostring(rels_root, encoding="utf-8", xml_declaration=True),
            )
            ET.register_namespace("", CT_NS)
            zout.writestr(
                "[Content_Types].xml",
                ET.tostring(content_types_root, encoding="utf-8", xml_declaration=True),
            )
            for media_name, image_path in media_to_add:
                if media_name in existing_names:
                    continue
                zout.write(image_path, media_name)


def print_dry_run(blocks: Iterable[Block]) -> None:
    counts: dict[str, int] = {}
    for block in blocks:
        counts[block.kind] = counts.get(block.kind, 0) + 1
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("markdown", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--backend",
        choices=("auto", "uno", "ooxml"),
        default="auto",
        help="Use LibreOffice UNO, pure OOXML fallback, or try UNO then fallback.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse Markdown and print block counts without using LibreOffice.",
    )
    args = parser.parse_args(argv)

    blocks = parse_markdown(args.markdown)
    if args.dry_run:
        print_dry_run(blocks)
        return 0

    if args.backend == "uno":
        build_docx(args.markdown, args.template, args.output)
    elif args.backend == "ooxml":
        build_docx_ooxml(args.markdown, args.template, args.output)
    else:
        try:
            build_docx(args.markdown, args.template, args.output)
        except Exception as exc:
            print(f"UNO backend failed; falling back to OOXML: {exc}", file=sys.stderr)
            build_docx_ooxml(args.markdown, args.template, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
