PDF_SRC = report.tex
PDF_OUT = Gradient\ Coding\ Report

pdf: clean $(PDF_SRC)
	pdflatex -halt-on-error -jobname=$(PDF_OUT) $(PDF_SRC) && pdflatex -halt-on-error -jobname=$(PDF_OUT) $(PDF_SRC) && make clean

clean:
	rm -f *.aux *.log