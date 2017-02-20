all::	notes.pdf

clean:
	-rm -f notes.pdf *.aux *.log *.bbl

notes.pdf:	notes.tex bib.bib
	pdflatex notes
	bibtex notes
	pdflatex notes
	pdflatex notes

