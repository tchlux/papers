FNAME = paper

all: $(FNAME).pdf
	rm -f $(FNAME).bbl $(FNAME).blg $(FNAME).log $(FNAME).aux

$(FNAME).pdf: $(FNAME).tex $(FNAME).bib
	pdflatex $(FNAME).tex
	bibtex $(FNAME)
	pdflatex $(FNAME).tex
	pdflatex $(FNAME).tex
	chmod 666 $(FNAME).pdf

clean:
	rm -f $(FNAME).bbl $(FNAME).blg $(FNAME).log $(FNAME).aux $(FNAME).pdf
