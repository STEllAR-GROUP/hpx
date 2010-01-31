 
PAPER = had

dvi:
	latex    $(PAPER)
	bibtex   $(PAPER)
	latex    $(PAPER)
	latex    $(PAPER)

pdf: dvi
#	dvipdf $(PAPER)
	dvips -tletter -o $(PAPER).ps $(PAPER).dvi
	ps2pdf $(PAPER).ps


ps: dvi
	dvips  $(PAPER)

clean:
	/bin/rm -f *.dvi *.log *.aux *.toc *.bbl *.blg 
	/bin/rm -f $(PAPER).ps $(PAPER).pdf
