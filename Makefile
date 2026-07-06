FIGURES=figure.pdf

all: supp.pdf paper.docx

# Supplementary information (kept as a LaTeX-built PDF).
supp.pdf: supp.tex authors.tex tools_table.tex functionality_table.tex paper.bib
	pdflatex supp.tex
	bibtex supp
	pdflatex supp.tex
	pdflatex supp.tex

# Main text. paper.md is the editable master (produced once from the former
# paper.tex by convert_to_markdown.py); the Word file for submission is a
# direct pandoc conversion of it. Citations and references are already inlined
# in paper.md, so no bibliography processing is needed here.
paper.docx: paper.md
	pandoc paper.md -o paper.docx

# Regenerate plot figures from data.
figures/%.pdf: plot.py
	python3 plot.py $*

# Rasterise/convert SVG figures to PDF.
%.pdf: %.svg
	inkscape $< --export-filename=$@

clean:
	rm -f *.log *.dvi *.aux
	rm -f *.blg *.bbl
	rm -f *.eps *.[1-9]
	rm -f src/*.mpx *.mpx

mrproper: clean
	rm -f *.ps *.pdf *.docx
