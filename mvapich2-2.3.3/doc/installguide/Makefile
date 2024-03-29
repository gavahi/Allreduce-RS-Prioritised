ALL: all-redirect
SHELL = /bin/sh
srcdir = .
MPICH_VERSION = 3.2.1

.SUFFIXES: .pdf .dvi .tex

# Set these to the location of the web and ftp locations for the documents
WEBDIR = /mcs/www/mpi/mpich/docs
FTPDIR = /home/ftp/pub/mpi/mpich

VPATH=.:${srcdir}

LATEX      = TEXINPUTS=".:$(srcdir):" ; export TEXINPUTS ; latex
BIBTEX     = BIBINPUTS=".:$(srcdir):" ; export BIBINPUTS ; bibtex
DVIPDFM    = TEXINPUTS=".:$(srcdir):" ; export TEXINPUTS ; dvipdf
LATEX2HTML = latex2html

# Update the %MPICH_VERSION% with current version string.
install.tex: install.tex.vin
	sed -e "s/%MPICH_VERSION%/${MPICH_VERSION}/g" $? > $@

install.dvi: install.tex
	-$(LATEX) install.tex
	-$(LATEX) install.tex
	$(LATEX)  install.tex

install.pdf: install.dvi
	$(DVIPDFM) install.dvi

# Use latex2html to create the HTML version.  There are several 
# steps needed to workaround features of latex2html
# 1. latex2html needs the install.aux file, so we run a latex step if
#    necessary
# 2. latex2html is not compatible with VPATH builds (it runs in the
#    directory that contains the source file), so we copy the source
#    file.  If more source files are added, this step will need to 
#    be changed
install.aux: install.tex
	$(LATEX) install.tex

install/install.html: install.aux install.tex
	$(LATEX2HTML) install.tex

# Here are some thoughts on using tohtml
#	tohtml -debugfile -default -dosnl -gaudy \
#              -basedef $(srcdir)/../mpiman.def $(srcdir)/install.tex
#	tohtml -default -dosnl -gaudy -basedef $(srcdir)/../mpiman.def \
#	$(srcdir)/install.tex

all-redirect: install.pdf

mandoc:

htmldoc:

latexdoc: ALL

clean:
	-rm -f *.dvi *.log *.out *.aux *.toc *.bbl *.blg *.pdf *.ps

distclean: clean

install: install.pdf install/install.html
	@if [ -z "$(WEBDIR)" ] ; then echo "No WEBDIR set" ; exit 1 ; fi
	@if [ ! -d $(WEBDIR) ] ; then echo "Create $(WEBDIR) first" ; exit 1 ;fi
	@if [ -z "$(FTPDIR)" ] ; then echo "No FTPDIR set" ; exit 1 ; fi
	@if [ ! -d $(FTPDIR) ] ; then echo "Create $(FTPDIR) first" ; exit 1 ;fi
	cp install.pdf $(WEBDIR)/install.pdf
	cp -rp install $(WEBDIR)/install-new
	rm -rf $(WEBDIR)/install
	mv -f $(WEBDIR)/install-new $(WEBDIR)/install
	cp install.pdf $(FTPDIR)/install.pdf
	cp -rp install $(FTPDIR)/install-new
	rm -rf $(WEBDIR)/install
	mv -f $(FTPDIR)/install-new $(FTPDIR)/install
