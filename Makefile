FIGURES=figure.pdf

all: paper.pdf supp.pdf review-diff.pdf

paper.pdf: paper.tex authors.tex paper.bib ${FIGURES}
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex

supp.pdf: supp.tex authors.tex tools_table.tex functionality_table.tex paper.bib
	pdflatex supp.tex
	bibtex supp
	pdflatex supp.tex
	pdflatex supp.tex

arxiv-submission.tar.gz:
	rm -fR arxiv-submission
	mkdir arxiv-submission
	cp paper.tex supp.tex authors.tex tools_table.tex functionality_table.tex naturemag.bst paper.bib figure.pdf ./arxiv-submission/
	tar -zcvf arxiv-submission.tar.gz arxiv-submission

review-diff.tex: paper.tex
	latexdiff reviewed-paper.tex paper.tex > review-diff.tex

review-diff.pdf: review-diff.tex
	pdflatex review-diff.tex
	pdflatex review-diff.tex
	bibtex review-diff
	pdflatex review-diff.tex


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

%.pdf: %.svg
	inkscape $< --export-filename=$@
