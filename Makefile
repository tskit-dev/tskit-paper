FIGURES=figure.pdf

paper.pdf: paper.tex authors.tex tools_table.tex paper.bib
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex

paper.ps: paper.dvi
	dvips paper

paper.dvi: paper.tex paper.bib
	latex paper.tex
	bibtex paper
	latex paper.tex
	latex paper.tex

figures/%.pdf: plot.py
	python3 plot.py $*

clean:
	rm -f *.log *.dvi *.aux
	rm -f *.blg *.bbl
	rm -f *.eps *.[1-9]
	rm -f src/*.mpx *.mpx

mrproper: clean
	rm -f *.ps *.pdf

%.pdf : %.svg
	inkscape $< --export-filename=$@
