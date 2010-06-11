# $Id  $

PYTHON:=python
CWM:=../cwm.py

DELITSUB = 'data:text/rdf+n3;charset=utf-8,%40prefix%20log%3A%20%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F10%2Fswap%2Flog%23%3E%20.%0D%0A%7B%20%3Fs%20%3Fp%20%3Fo%3B%20log%3ArawType%20%5B%20log%3AnotEqualTo%20log%3ALiteral%20%5D%20%7D%20%0D%0A%20%20%20%3D%3E%20%7B%20%3Fs%20%3Fp%20%3Fo%20%7D%20.'

sparql_table.py: table_generator.py sparql-selectors.rdf
	$(PYTHON) table_generator.py --grammar=sparql-selectors.rdf sparql_table.py
sparql-selectors.rdf: %.rdf: %.n3 
	$(PYTHON) $(CWM) --n3 $< --filter=$(DELITSUB) --rdf > $@

sparql-selectors.n3: bnf-token-rules.n3
	$(PYTHON) $(CWM) http://www.w3.org/2000/10/swap/grammar/sparql bnf-token-rules.n3 --think --purge --data > $@

.PHONY : clean

clean:
	rm *.rdf sparql-selectors.n3 sparql_table.py
