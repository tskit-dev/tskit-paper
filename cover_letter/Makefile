cover_letter.pdf: cover_letter.tex
	pdflatex cover_letter.tex
	bibtex cover_letter
	pdflatex cover_letter.tex
	pdflatex cover_letter.tex

clean:
	rm -f *.log *.dvi *.aux
	rm -f *.blg *.bbl
	rm -f src/*.mpx *.mpx
	rm *.pdf

