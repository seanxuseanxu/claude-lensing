# pdflatex is not on the default PATH on this cluster:
#     module load texlive/2024
# Figures are pulled from ../model/runs/bulgedisk_long/ via \graphicspath,
# not copied here, so the report always reflects the run it describes.

desi238.pdf: desi238.tex
	pdflatex -interaction=nonstopmode -halt-on-error desi238.tex
	pdflatex -interaction=nonstopmode -halt-on-error desi238.tex

clean:
	rm -f desi238.aux desi238.log desi238.out desi238.toc

.PHONY: clean
