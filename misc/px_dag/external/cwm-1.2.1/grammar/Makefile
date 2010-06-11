#  Build stuff up from the CFG

SWAP=..
P=python
C=$(SWAP)/cwm.py
CG=PYTHONPATH=$(SWAP) python predictiveParser.py


.SUFFIXES: .rdf .n3

all : n3-checked.txt n3-report.html n3.rdf \
		n3rdf-report.html \
		n3rules-report.html \
		n3ql-report.html
	echo all done


#  N3 Full:

.n3.rdf:
	python $(SWAP)/cwm.py --n3 $< --rdf  > $@

n3-yacc.c : n3-selectors.n3-yacc.y
	yacc -o $@ n3-selectors.n3-yacc.y

n3-selectors.n3-yacc.y : n3-selectors.n3

n3-checked.txt: n3-selectors.n3
	$(CG) --grammar=n3-selectors.n3 \
		--as=http://www.w3.org/2000/10/swap/grammar/n3#document \
		--parse=n3.n3 && touch $@

n3-selectors.n3 : n3.n3 bnf-rules.n3
	time $P $C n3.n3 bnf-rules.n3 --think --purge --data --quiet > $@

n3-i18n-selectors.n3 : n3-re.n3 bnf-rules.n3
	time $P $C n3-re.n3 bnf-rules.n3 --think --purge --data --quiet > $@

n3-report.big:  n3.n3 bnf2html.n3
	time $P $C n3.n3 bnf2html.n3 --chatty=21 --think --data --quiet > $@

n3-report.html : n3-report.big
	time $P $C  n3-report.big --strings > $@

n3.rdf : n3.n3 
	$C --n3 n3.n3  --rdf > $@

# N3-RDF

n3rdf-checked.txt: n3rdf-selectors.n3
	$(CG) n3rdf-selectors.n3 \
		http://www.w3.org/2000/10/swap/grammar/n3rdf#document > $@

n3rdf-selectors.n3 : n3-rdf.n3 bnf-rules.n3
	time $P $C n3-rdf.n3 bnf-rules.n3 --think --purge --data > $@

n3rdf-report.big:  n3-rdf.n3 bnf2html.n3
	time $P $C n3-rdf.n3 bnf2html.n3 --chatty=21 --think --data > $@


n3rdf-report.html : n3rdf-report.big
	time $P $C  n3rdf-report.big --strings > $@

# N3-Rules

n3rules-selectors.n3 : n3-rules.n3 bnf-rules.n3
	time $P $C n3-rules.n3 bnf-rules.n3 --think --purge --data > $@

n3rules-report.big:  n3-rules.n3 bnf2html.n3
	time $P $C n3-rules.n3 bnf2html.n3 --chatty=21 --think --data > $@


n3rules-report.html : n3rules-report.big
	time $P $C  n3rules-report.big --strings > $@

# N3-QL

n3ql-checked.txt: n3ql-selectors.n3
	$(CG) n3ql-selectors.n3 \
		http://www.w3.org/2000/10/swap/grammar/n3ql#document > $@

n3ql-selectors.n3 : n3-ql.n3 bnf-rules.n3
	time $P $C n3-ql.n3 bnf-rules.n3 --think --purge --data > $@

n3ql-report.big:  n3-ql.n3 bnf2html.n3
	time $P $C n3-ql.n3 bnf2html.n3 --chatty=21 --think --data > $@


n3ql-report.html : n3ql-report.big
	time $P $C  n3ql-report.big --strings > $@

#___________________________________

Q=../../../../2001/sw/DataAccess/rq23
sparql.n3 : $Q/grammar.yy
	sed -f yy2n3.sed < $Q/grammar.yy > sparql.n3

sparql-utf16.n3 : sparql.n3 to-utf18.sed
	sed -f to-utf16.sed < sparql.n3 > sparql-utf16.n3

sparql-checked.txt: sparql-selectors.n3
	$(CG) --grammar=sparql-selectors.n3 \
		--as=http://www.w3.org/2000/10/swap/grammar/sparql#Query > $@

sparql-selectors.n3 : sparql.n3 bnf-rules.n3 shorthand-rules.n3
	time $P $C sparql.n3 shorthand-rules.n3 bnf-rules.n3 --think --purge --data > $@

sparql-report.big:  sparql.n3 bnf2html.n3
	time $P $C sparql.n3 bnf2html.n3 --chatty=21 --think --data > $@


sparql-report.html : sparql-report.big
	time $P $C  sparql-report.big --strings > $@


ebnf: notation3.rdf ebnf.rdf notation3.json turtle_syntax

turtle_syntax: turtle.json turtle-bnf.n3 turtle.n3 turtle.bnf

CHATTY=0

.SUFFIXES: .json .n3 .bnf


LYNX=elinks
turtle.bnf: turtle.html
	$(LYNX) -no-references -no-numbering -dump turtle.html >$@

turtle.json: turtle-bnf.n3 gramLL1.py
	PYTHONPATH=$(HOME)/lib/python:../.. $P gramLL1.py turtle-bnf.n3 'http://www.w3.org/2000/10/swap/grammar/turtle#language' >$@

turtle-bnf.n3: turtle.n3 ebnf2bnf.n3
	$P $C turtle.n3 ebnf2bnf.n3  --chatty=$(CHATTY) \
		--think --data >$@

turtle.n3: turtle.bnf ebnf2turtle.py
	$P ebnf2turtle.py $< ttl language 'http://www.w3.org/2000/10/swap/grammar/turtle#' >$@

notation3.json: notation3-bnf.n3 gramLL1.py
	PYTHONPATH=$(HOME)/lib/python:../.. $P gramLL1.py notation3-bnf.n3 'http://www.w3.org/2000/10/swap/grammar/notation3#language' >$@

notation3-ll1.n3: notation3.n3 ebnf2bnf.n3 first_follow.n3
	$P $C notation3.n3 ebnf2bnf.n3 first_follow.n3 --chatty=$(CHATTY) \
		--think --data >$@

notation3-bnf.n3: notation3.n3 ebnf2bnf.n3 first_follow.n3
	$P $C notation3.n3 ebnf2bnf.n3  --chatty=$(CHATTY) \
		--think --data >$@

notation3.n3: notation3.bnf ebnf2turtle.py
	$P ebnf2turtle.py $< n3 language 'http://www.w3.org/2000/10/swap/grammar/notation3#' >$@

ebnf.rdf: ebnf.n3
notation3.rdf: notation3.n3


#ends

