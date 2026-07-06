"""
One-shot, exact conversion of the main-text LaTeX (paper.tex) into a single
self-contained Markdown file (paper.md) that becomes the editable master.

The output is self-contained:
  * title, author list and affiliations are inlined as plain text (parsed from
    authors.tex, since authblk macros do not survive pandoc);
  * citations are inlined as superscript numbers with a numbered reference list
    (pandoc + citeproc + nature.csl), so no .bib is needed downstream;
  * the figure is reduced to its legend text ("**Figure 1.** ...") with no image,
    as the figure is submitted separately.

Run once, review paper.md, then the LaTeX sources/tooling can be removed.

Usage: python3 convert_to_markdown.py   # writes paper.md
"""
import re
import subprocess

TITLE = "Population-scale Ancestral Recombination Graphs with tskit 1.0"

# LaTeX joint-author markers -> unicode/escaped-markdown equivalents.
SYMBOLS = {
    r"$\ast$": r"\*",    # escaped so markdown does not read it as emphasis
    r"$\dagger$": "†",   # dagger
    r"$\ddagger$": "‡",  # double dagger
}

AUTHOR_RE = re.compile(r"\\author\[(.*?)\]\{(.*?)\}")
AFFIL_RE = re.compile(r"\\affil\[(.*?)\]\{(.*?)\}")

# Matches the pandoc-rendered figure image (implicit figure), with optional
# extension and trailing attribute block, so we can replace it with plain text.
FIGURE_RE = re.compile(
    r"!\[(?P<cap>.*?)\]\(figure[^)]*\)(?:\{[^}]*\})?",
    re.DOTALL,
)


def norm_label(label):
    return SYMBOLS.get(label.strip(), label.strip())


def build_front_matter():
    with open("authors.tex") as f:
        text = f.read()

    lines = [f"**{TITLE}**", ""]

    authors = []
    for labels, name in AUTHOR_RE.findall(text):
        marks = [norm_label(p) for p in labels.split(",")]
        authors.append(f"{name.strip()}^{','.join(marks)}^")
    lines.append(", ".join(authors))
    lines.append("")

    legend = []
    for label, body in AFFIL_RE.findall(text):
        label = label.strip()
        if label in SYMBOLS:
            legend.append(f"{SYMBOLS[label]}{body.strip()}")
        else:
            lines.append(f"^{label}^{body.strip()}  ")  # trailing spaces = break
    lines.append("")
    if legend:
        lines.append("; ".join(legend))
        lines.append("")

    return "\n".join(lines)


def build_body():
    out = subprocess.run(
        [
            "pandoc", "paper.tex",
            "--to=markdown-citations",
            "--citeproc",
            "--bibliography=paper.bib",
            "--csl=nature.csl",
            "--wrap=none",
            "--metadata", "reference-section-title=References",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    body = out.stdout

    # Replace the figure image with its legend text only (no embedded image).
    def repl(m):
        return f"**Figure 1.** {m.group('cap').strip()}"

    body, n = FIGURE_RE.subn(repl, body)
    if n != 1:
        raise SystemExit(f"expected exactly one figure image, replaced {n}")

    # Flatten citeproc's CSL bibliography markup into a plain numbered list:
    # "[1. ]{.csl-left-margin}[Text]{.csl-right-inline}" -> "1. Text".
    body = re.sub(
        r"\[(\d+)\.\s*\]\{\.csl-left-margin\}\[(.*?)\]\{\.csl-right-inline\}",
        r"\1. \2",
        body,
    )
    # Drop the pandoc fenced-div wrappers (::: {#refs ...}, ::: {#ref-... }, :::).
    body = "\n".join(
        ln for ln in body.split("\n") if not ln.lstrip().startswith(":::")
    )
    # Strip auto-generated heading attributes ({#id .unnumbered}) and normalise
    # the References heading to match the other section headings.
    body = re.sub(r"^(#{1,6} .+?)\s*\{[^}]*\}\s*$", r"\1", body, flags=re.M)
    body = re.sub(r"^# References$", "## References", body, flags=re.M)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body


def main():
    front = build_front_matter()
    body = build_body()
    with open("paper.md", "w") as f:
        f.write(front.rstrip() + "\n\n" + body.strip() + "\n")


if __name__ == "__main__":
    main()
