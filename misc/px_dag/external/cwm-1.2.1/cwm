#!/usr/bin/python
"""
$Id: cwm.py,v 1.197 2007/12/13 15:38:39 syosi Exp $

Closed World Machine

(also, in Wales, a valley  - topologiclly a partially closed world perhaps?)

This is an application which knows a certian amount of stuff and can manipulate
it. It uses llyn, a (forward chaining) query engine, not an (backward chaining)
inference engine: that is, it will apply all rules it can but won't figure out
which ones to apply to prove something. 


License
-------
Cwm: http://www.w3.org/2000/10/swap/doc/cwm.html

Copyright (c) 2000-2004 World Wide Web Consortium, (Massachusetts 
Institute of Technology, European Research Consortium for Informatics 
and Mathematics, Keio University). All Rights Reserved. This work is 
distributed under the W3C Software License [1] in the hope that it 
will be useful, but WITHOUT ANY WARRANTY; without even the implied 
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

[1] http://www.w3.org/Consortium/Legal/2002/copyright-software-20021231

"""

#the following lines should be removed. They will NOT work with any distribution
#-----------------
from os import chdir, getcwd
from sys import path
qqq = getcwd()
chdir(path[0])
chdir('..')
path.append(getcwd())
chdir(qqq)
#import swap
#print dir(swap)
#-----------------
#end lines should be removed


import string, sys

# From  http://www.w3.org/2000/10/swap/
from swap import  diag
from swap.why import  explainFormula, newTopLevelFormula
from swap.diag import verbosity, setVerbosity, progress, tracking, setTracking
from swap.uripath import join, splitFrag
from swap.webAccess import urlopenForRDF, load, sandBoxed 

from swap import  notation3            # N3 parsers and generators
from swap import  toXML                 #  RDF generator

from swap.why import BecauseOfCommandLine
from swap.query import think, applyRules, applyQueries, applySparqlQueries, testIncludes
from swap.update import patch

from swap import  uripath
from swap import  llyn
from swap import  RDFSink

cvsRevision = "$Revision: 1.197 $"
    
            

#################################################  Command line

    
def doCommand():
        """Command line RDF/N3 tool
        
 <command> <options> <steps> [--with <more args> ]

options:
 
--pipe        Don't store, just pipe out *

steps, in order left to right:

--rdf         Input & Output ** in RDF/XML insead of n3 from now on
--n3          Input & Output in N3 from now on. (Default)
--rdf=flags   Input & Output ** in RDF and set given RDF flags
--n3=flags    Input & Output in N3 and set N3 flags
--ntriples    Input & Output in NTriples (equiv --n3=usbpartane -bySubject -quiet)
--language=x  Input & Output in "x" (rdf, n3, etc)  --rdf same as: --language=rdf
--languageOptions=y     --n3=sp same as:  --language=n3 --languageOptions=sp
--ugly        Store input and regurgitate, data only, fastest *
--bySubject   Store input and regurgitate in subject order *
--no          No output *
              (default is to store and pretty print with anonymous nodes) *
--base=<uri>  Set the base URI. Input or output is done as though theis were the document URI.
--closure=flags  Control automatic lookup of identifiers (see below)
<uri>         Load document. URI may be relative to current directory.

--apply=foo   Read rules from foo, apply to store, adding conclusions to store
--patch=foo   Read patches from foo, applying insertions and deletions to store
--filter=foo  Read rules from foo, apply to store, REPLACING store with conclusions
--query=foo   Read a N3QL query from foo, apply it to the store, and replace the store with its conclusions
--sparql=foo   Read a SPARQL query from foo, apply it to the store, and replace the store with its conclusions
--rules       Apply rules in store to store, adding conclusions to store
--think       as -rules but continue until no more rule matches (or forever!)
--engine=otter use otter (in your $PATH) instead of llyn for linking, etc
--why         Replace the store with an explanation of its contents
--why=u       proof tries to be shorter
--mode=flags  Set modus operandi for inference (see below)
--reify       Replace the statements in the store with statements describing them.
--dereify     Undo the effects of --reify
--flatten     Reify only nested subexpressions (not top level) so that no {} remain.
--unflatten   Undo the effects of --flatten
--think=foo   as -apply=foo but continue until no more rule matches (or forever!)
--purge       Remove from store any triple involving anything in class log:Chaff
--data              Remove all except plain RDF triples (formulae, forAll, etc)
--strings     Dump :s to stdout ordered by :k whereever { :k log:outputString :s }
--crypto      Enable processing of crypto builtin functions. Requires python crypto.
--help        print this message
--revision    print CVS revision numbers of major modules
--chatty=50   Verbose debugging output of questionable use, range 0-99
--sparqlServer instead of outputting, start a SPARQL server on port 8000 of the store
--sparqlResults        After sparql query, print in sparqlResults format instead of rdf

finally:

--with        Pass any further arguments to the N3 store as os:argv values
 

            * mutually exclusive
            ** doesn't work for complex cases :-/
Examples:
  cwm --rdf foo.rdf --n3 --pipe     Convert from rdf/xml to rdf/n3
  cwm foo.n3 bar.n3 --think         Combine data and find all deductions
  cwm foo.n3 --flat --n3=spart

Mode flags affect inference extedning to the web:
 r   Needed to enable any remote stuff.
 a   When reading schema, also load rules pointed to by schema (requires r, s)
 E   Errors loading schemas of definitive documents are ignored
 m   Schemas and definitive documents laoded are merged into the meta knowledge
     (otherwise they are consulted independently)
 s   Read the schema for any predicate in a query.
 u   Generate unique ids using a run-specific

Closure flags are set to cause the working formula to be automatically exapnded to
the closure under the operation of looking up:

 s   the subject of a statement added
 p   the predicate of a statement added
 o   the object of a statement added
 t   the object of an rdf:type statement added
 i   any owl:imports documents
 r   any doc:rules documents
 E   errors are ignored --- This is independant of --mode=E

 n   Normalize IRIs to URIs
 e   Smush together any nodes which are = (owl:sameAs)

See http://www.w3.org/2000/10/swap/doc/cwm  for more documentation.

Setting the environment variable CWM_RDFLIB to 1 maked Cwm use rdflib to parse
rdf/xml files. Note that this requires rdflib.
"""
        
        import time
        import sys
        from swap import  myStore

        # These would just be attributes if this were an object
        global _store
        global workingContext
        option_need_rdf_sometime = 0  # If we don't need it, don't import it
                               # (to save errors where parsers don't exist)
        
        option_pipe = 0     # Don't store, just pipe though
        option_inputs = []
        option_reify = 0    # Flag: reify on output  (process?)
        option_flat = 0    # Flag: reify on output  (process?)
        option_crypto = 0  # Flag: make cryptographic algorithms available
        setTracking(0)
        option_outURI = None
        option_outputStyle = "-best"
        _gotInput = 0     #  Do we not need to take input from stdin?
        option_meta = 0
        option_normalize_iri = 0
        
        option_flags = { "rdf":"l", "n3":"", "think":"", "sparql":""}
            # RDF/XML serializer can't do list ("collection") syntax.
            
        option_quiet = 0
        option_with = None  # Command line arguments made available to N3 processing
        option_engine = "llyn"
        option_why = ""
        
        _step = 0           # Step number used for metadata
        _genid = 0

        hostname = "localhost" # @@@@@@@@@@@ Get real one
        
        # The base URI for this process - the Web equiv of cwd
        _baseURI = uripath.base()
        
        option_format = "n3"      # set the default format
        option_first_format = None
        
        _outURI = _baseURI
        option_baseURI = _baseURI     # To start with - then tracks running base
        
        #  First pass on command line        - - - - - - - P A S S  1
        
        for argnum in range(1,len(sys.argv)):  # options after script name
            arg = sys.argv[argnum]
            if arg.startswith("--"): arg = arg[1:]   # Chop posix-style -- to -
#            _equals = string.find(arg, "=")
            _lhs = ""
            _rhs = ""
            try:
                [_lhs,_rhs]=arg.split('=',1)
                try:
                    _uri = join(option_baseURI, _rhs)
                except ValueError:
                    _uri = _rhs
            except ValueError: pass
            if arg == "-ugly": option_outputStyle = arg
            elif _lhs == "-base": option_baseURI = _uri
            elif arg == "-rdf":
                option_format = "rdf"
                if option_first_format == None:
                    option_first_format = option_format 
                option_need_rdf_sometime = 1
            elif _lhs == "-rdf":
                option_format = "rdf"
                if option_first_format == None:
                    option_first_format = option_format 
                option_flags["rdf"] = _rhs
                option_need_rdf_sometime = 1
            elif arg == "-n3":
                option_format = "n3"
                if option_first_format == None:
                    option_first_format = option_format 
            elif _lhs == "-n3":
                option_format = "n3"
                if option_first_format == None:
                    option_first_format = option_format 
                option_flags["n3"] = _rhs
            elif _lhs == "-mode":
                option_flags["think"] = _rhs
            elif _lhs == "-closure":
                if "n" in _rhs:
                    option_normalize_iri = 1
            #elif _lhs == "-solve":
            #    sys.argv[argnum+1:argnum+1] = ['-think', '-filter=' + _rhs]
            elif _lhs == "-language":
                option_format = _rhs
                if option_first_format == None:
                    option_first_format = option_format
            elif _lhs == "-languageOptions":
                option_flags[option_format] = _rhs
            elif arg == "-quiet": option_quiet = 1
            elif arg == "-pipe": option_pipe = 1
            elif arg == "-crypto": option_crypto = 1
            elif _lhs == "-why":
                diag.tracking=1
                diag.setTracking(1)
                option_why = _rhs
            elif arg == "-why":
                diag.tracking=1
                diag.setTracking(1)
                option_why = ""
            elif arg == "-track":
                diag.tracking=1
                diag.setTracking(1)
            elif arg == "-bySubject": option_outputStyle = arg
            elif arg == "-no": option_outputStyle = "-no"
            elif arg == "-debugString": option_outputStyle = "-debugString"
            elif arg == "-strings": option_outputStyle = "-no"
            elif arg == "-sparqlResults": option_outputStyle = "-no"
            elif arg == "-triples" or arg == "-ntriples":
                option_format = "n3"
                option_flags["n3"] = "bravestpun"
                option_outputStyle = "-bySubject"
                option_quiet = 1
            elif _lhs == "-outURI": option_outURI = _uri
            elif _lhs == "-chatty":
                setVerbosity(int(_rhs))
            elif arg[:7] == "-apply=": pass
            elif arg[:7] == "-patch=": pass
            elif arg == "-reify": option_reify = 1
            elif arg == "-flat": option_flat = 1
            elif arg == "-help":
                print doCommand.__doc__
                print notation3.ToN3.flagDocumentation
                print toXML.ToRDF.flagDocumentation
                try:
                    from swap import  sax2rdf      # RDF1.0 syntax parser to N3 RDF stream
                    print sax2rdf.RDFXMLParser.flagDocumentation
                except:
                    pass
                return
            elif arg == "-revision":
                progress( "cwm=",cvsRevision, "llyn=", llyn.cvsRevision)
                return
            elif arg == "-with":
                option_with = sys.argv[argnum+1:] # The rest of the args are passed to n3
                break
            elif arg[0] == "-": pass  # Other option
            else :
                option_inputs.append(join(option_baseURI, arg))
                _gotInput = _gotInput + 1  # input filename
            

        # Between passes, prepare for processing
        setVerbosity(0)

        if not option_normalize_iri:
            llyn.canonical = lambda x: x

        #  Base defauts
        if option_baseURI == _baseURI:  # Base not specified explicitly - special case
            if _outURI == _baseURI:      # Output name not specified either
                if _gotInput == 1:  # But input file *is*, 
                    _outURI = option_inputs[0]        # Just output to same URI
                    option_baseURI = _outURI          # using that as base.
                if diag.tracking:
                    _outURI = RDFSink.runNamespace()[:-1]
                    option_baseURI = _outURI
        option_baseURI = splitFrag(option_baseURI)[0]

        #  Fix the output sink
        if option_format == "rdf":
            _outSink = toXML.ToRDF(sys.stdout, _outURI, base=option_baseURI,                                                 flags=option_flags["rdf"])
        elif option_format == "n3" or option_format == "sparql":
            _outSink = notation3.ToN3(sys.stdout.write, base=option_baseURI,
                                      quiet=option_quiet, flags=option_flags["n3"])
        elif option_format == "trace":
            _outSink = RDFSink.TracingRDFSink(_outURI, base=option_baseURI,
                        flags=option_flags.get("trace",""))
            if option_pipe:
                # this is really what a parser wants to dump to
                _outSink.backing = llyn.RDFStore( _outURI+"#_g",
                    argv=option_with, crypto=option_crypto) 
            else:
                # this is really what a store wants to dump to 
                _outSink.backing = notation3.ToN3(sys.stdout.write,
                        base=option_baseURI, quiet=option_quiet,
                        flags=option_flags["n3"])

                #  hm.  why does TimBL use sys.stdout.write, above?  performance at the                
        else:
            raise NotImplementedError

        version = "$Id: cwm.py,v 1.197 2007/12/13 15:38:39 syosi Exp $"
        if not option_quiet and option_outputStyle != "-no":
            _outSink.makeComment("Processed by " + version[1:-1]) # Strip $ to disarm
            _outSink.makeComment("    using base " + option_baseURI)

        if option_flat:
            _outSink = notation3.Reifier(_outSink, _outURI+ "#_formula", flat=1)

        if diag.tracking: 
            myReason = BecauseOfCommandLine(`sys.argv`)
            # @@ add user, host, pid, pwd, date time? Privacy!
        else:
            myReason = None

        if option_pipe:
            _store = _outSink
            workingContext = _outSink #.newFormula()
        else:
            if "u" in option_flags["think"]:
                _store = llyn.RDFStore(argv=option_with, crypto=option_crypto)
            else:
                _store = llyn.RDFStore( _outURI+"#_g",
                                        argv=option_with, crypto=option_crypto)
            myStore.setStore(_store)


            if  _gotInput: 
                workingContext = _store.newFormula(option_inputs [0]+"#_work")
                newTopLevelFormula(workingContext)
            else: # default input
                if option_first_format is None: option_first_format = option_format
                ContentType={ "rdf": "application/xml+rdf", "n3":
                                    "text/rdf+n3", "sparql":
                              "x-application/sparql"}[option_first_format]
                workingContext = _store.load(
    #                            asIfFrom = join(_baseURI, ".stdin"),
                                asIfFrom = _baseURI,
                                contentType = ContentType,
                                flags = option_flags[option_first_format],
                                remember = 0,
                                referer = "",
                                why = myReason, topLevel=True)
                workingContext.reopen()
        workingContext.stayOpen = 1 # Never canonicalize this. Never share it.
        

        # ____________________________________________________________________
        #  Take commands from command line:- - - - - P A S S 2

        option_format = "n3"      # Use RDF/n3 rather than RDF/XML 
        option_flags = { "rdf":"l", "n3":"", "think": "", "sparql":"" } 
        option_quiet = 0
        _outURI = _baseURI
        option_baseURI = _baseURI     # To start with
        
        def filterize():
            """implementation of --filter
            for the --filter command, so we don't have it printed twice
            """
            global workingContext
            global r
            workingContext = workingContext.canonicalize()
            _store._formulaeOfLength = {}
            filterContext = _store.newFormula()
            newTopLevelFormula(filterContext)
            _store.load(_uri, openFormula=filterContext,
                                        why=myReason, referer="")
            _newContext = _store.newFormula()
            newTopLevelFormula(_newContext)
            applyRules(workingContext, filterContext, _newContext)
            workingContext.close()
            workingContext = _newContext

        sparql_query_formula = None

                
        for arg in sys.argv[1:]:  # Command line options after script name
            if verbosity()>5: progress("Processing %s." % (arg))
            if arg.startswith("--"): arg = arg[1:]   # Chop posix-style -- to -
            _equals = string.find(arg, "=")
            _lhs = ""
            _rhs = ""
            if _equals >=0:
                _lhs = arg[:_equals]
                _rhs = arg[_equals+1:]
            try:
                _uri = join(option_baseURI, _rhs)
            except ValueError:
                _uri =_rhs
            if arg[0] != "-":
                _inputURI = join(option_baseURI, splitFrag(arg)[0])
                assert ':' in _inputURI
                ContentType={ "rdf": "application/xml+rdf", "n3":
                                "text/rdf+n3",
                              "sparql": "x-application/sparql"}[option_format]

                if not option_pipe: workingContext.reopen()
                try:
                    load(_store, _inputURI,
                                openFormula=workingContext,
                                contentType =ContentType,
                                flags=option_flags[option_format],
                                referer="",
                            why=myReason)
                except:
                    progress(_inputURI)
                    raise

                _gotInput = 1

            elif arg == "-help":
                pass  # shouldn't happen
            elif arg == "-revision":
                pass
            elif _lhs == "-base":
                option_baseURI = _uri
                if verbosity() > 10: progress("Base now "+option_baseURI)

            elif arg == "-ugly":
                option_outputStyle = arg            

            elif arg == "-crypto": pass
            elif arg == "-pipe": pass
            elif _lhs == "-outURI": option_outURI = _uri

            elif arg == "-rdf": option_format = "rdf"
            elif _lhs == "-rdf":
                option_format = "rdf"
                option_flags["rdf"] = _rhs
            elif _lhs == "-mode":
                option_flags["think"] = _rhs
            elif _lhs == "-closure":
                workingContext.setClosureMode(_rhs)
            elif arg == "-n3": option_format = "n3"
            elif _lhs == "-n3":
                option_format = "n3"
                option_flags["n3"] = _rhs
            elif _lhs == "-language":
                option_format = _rhs
                if option_first_format == None:
                    option_first_format = option_format
            elif _lhs == "-languageOptions":
                option_flags[option_format] = _lhs
            elif arg == "-quiet" : option_quiet = 1            
            elif _lhs == "-chatty": setVerbosity(int(_rhs))
            elif arg[:7] == "-track=":
                diag.tracking = int(_rhs)
                
            elif option_pipe: ############## End of pipable options
                print "# Command line error: %s illegal option with -pipe", arg
                break

            elif arg == "-triples" or arg == "-ntriples":
                option_format = "n3"
                option_flags["n3"] = "spartan"
                option_outputStyle = "-bySubject"
                option_quiet = 1

            elif arg == "-bySubject":
                option_outputStyle = arg

            elif arg == "-debugString":
                option_outputStyle = arg

            elif arg[:7] == "-apply=":
                workingContext = workingContext.canonicalize()
                
                filterContext = _store.load(_uri, 
                            flags=option_flags[option_format],
                            referer="",
                            why=myReason, topLevel=True)
                workingContext.reopen()
                applyRules(workingContext, filterContext);

            elif arg[:7] == "-apply=":
                workingContext = workingContext.canonicalize()
                
                filterContext = _store.load(_uri, 
                            flags=option_flags[option_format],
                            referer="",
                            why=myReason, topLevel=True)
                workingContext.reopen()
                applyRules(workingContext, filterContext);

            elif arg[:7] == "-patch=":
                workingContext = workingContext.canonicalize()
                
                filterContext = _store.load(_uri, 
                            flags=option_flags[option_format],
                            referer="",
                            why=myReason, topLevel=True)
                workingContext.reopen()
                patch(workingContext, filterContext);

            elif _lhs == "-filter":
                filterize()

            elif _lhs == "-query":
                workingContext = workingContext.canonicalize()
                filterContext = _store.load(_uri, 
                            flags=option_flags[option_format],
                            referer="",
                            why=myReason, topLevel=True)
                _newContext = _store.newFormula()
                applyQueries(workingContext, filterContext, _newContext)
                workingContext.close()
                workingContext = _newContext

            elif _lhs == "-sparql":
                workingContext.stayOpen = False
                workingContext = workingContext.canonicalize()
                filterContext = _store.load(_uri, why=myReason,
                            referer="", contentType="x-application/sparql")
                _newContext = _store.newFormula()
                _newContext.stayOpen = True
                sparql_query_formula = filterContext
                applySparqlQueries(workingContext, filterContext, _newContext)
#                workingContext.close()
                workingContext = _newContext

            elif _lhs == "-why" or arg == "-why":
                workingContext.stayOpen = False
                workingContext = workingContext.close()
                workingContext = explainFormula(workingContext, option_why)
                # Can't prove proofs
                diag.tracking=0
                diag.setTracking(0)

            elif arg == "-dump":
                
                workingContext = workingContext.canonicalize()
                progress("\nDump of working formula:\n" + workingContext.debugString())
                
            elif arg == "-purge":
                workingContext.reopen()
                _store.purge(workingContext)
                
            elif arg == "-purge-rules" or arg == "-data":
                
                workingContext.reopen()
                _store.purgeExceptData(workingContext)

            elif arg == "-rules":
                
                workingContext.reopen()
                applyRules(workingContext, workingContext)

            elif arg[:7] == "-think=":
                
                filterContext = _store.load(_uri, referer="", why=myReason, topLevel=True)
                if verbosity() > 4:
                    progress( "Input rules to --think from " + _uri)
                workingContext.reopen()
                think(workingContext, filterContext, mode=option_flags["think"])

            elif arg[:7] == "-solve=":
                # --solve is a combination of --think and --filter.
                think(workingContext, mode=option_flags["think"])
                filterize()
                
            elif _lhs == "-engine":
                option_engine = _rhs
                
            elif arg == "-think":
                workingContext.isWorkingContext = True
                think(workingContext, mode=option_flags["think"])

            elif arg == '-rete':
                from swap import pycwmko                
                pythink = pycwmko.directPychinkoQuery(workingContext)
                #return
                #pythink()
                """
                    from pychinko import interpreter
                    from swap.set_importer import Set, ImmutableSet
                    pyf = pycwmko.N3Loader.N3Loader()
                    conv = pycwmko.ToPyStore(pyf)
                    conv.statements(workingContext)
                    interp = interpreter.Interpreter(pyf.rules[:])
                    interp.addFacts(Set(pyf.facts), initialSet=True)
                    interp.run()
                    pyf.facts = interp.totalFacts
                    workingContext = workingContext.store.newFormula()
                    reconv = pycwmko.FromPyStore(workingContext, pyf)
                    reconv.run()
                """

            elif arg == '-sparqlServer':
                from swap.sparql import webserver
                from swap import cwm_sparql
                sandBoxed(True)
                workingContext.stayOpen = False
                workingContext = workingContext.canonicalize()
                def _handler(s):
                    return cwm_sparql.sparql_queryString(workingContext, s)
                webserver.sparql_handler = _handler
                webserver.run()

            elif arg == "-lxkbdump":  # just for debugging
                raise NotImplementedError

            elif arg == "-lxfdump":   # just for debugging
                raise NotImplementedError               

            elif _lhs == "-prove":

                # code copied from -filter without really being understood  -sdh
                _tmpstore = llyn.RDFStore( _outURI+"#_g", metaURI=_metaURI, argv=option_with, crypto=option_crypto)

                tmpContext = _tmpstore.newFormula(_uri+ "#_formula")
                _newURI = join(_baseURI, "_w_"+`_genid`)  # Intermediate
                _genid = _genid + 1
                _newContext = _tmpstore.newFormula(_newURI+ "#_formula")
                _tmpstore.loadURI(_uri)

                print targetkb

            elif arg == "-flatten":
                #raise NotImplementedError
                from swap import reify
                workingContext = reify.flatten(workingContext)

            elif arg == "-unflatten":
                from swap import reify
                workingContext = reify.unflatten(workingContext)
                #raise NotImplementedError
                
            elif arg == "-reify":
                from swap import reify
                workingContext = reify.reify(workingContext)
                

            elif arg == "-dereify":
                from swap import reify
                workingContext = reify.dereify(workingContext)                
                

            elif arg == "-size":
                progress("Size: %i statements in store, %i in working formula."
                    %(_store.size, workingContext.size()))

            elif arg == "-strings":  # suppress output
                workingContext.outputStrings() 
                option_outputStyle = "-no"

            elif arg == '-sparqlResults':
                from cwm_sparql import outputString, SPARQL_NS
                ns = _store.newSymbol(SPARQL_NS)
                if not sparql_query_formula:
                    raise ValueError('No query')
                else:
                    sys.stdout.write(outputString(sparql_query_formula, workingContext)[0].encode('utf_8'))
                    option_outputStyle = "-no"
                    
                
            elif arg == "-no":  # suppress output
                option_outputStyle = arg
                
            elif arg[:8] == "-outURI=": pass
            elif arg == "-with": break
            else:
                progress( "cwm: Unknown option: " + arg)
                sys.exit(-1)



        # Squirt it out if not piped

        workingContext.stayOpen = 0  # End its use as an always-open knoweldge base
        if option_pipe:
            workingContext.endDoc()
        else:
            if hasattr(_outSink, "serializeKB"):
                raise NotImplementedError
            else:
                if verbosity()>5: progress("Begining output.")
                workingContext = workingContext.close()
                assert workingContext.canonical != None

                if option_outputStyle == "-ugly":
                    _store.dumpChronological(workingContext, _outSink)
                elif option_outputStyle == "-bySubject":
                    _store.dumpBySubject(workingContext, _outSink)
                elif option_outputStyle == "-no":
                    pass
                elif option_outputStyle == "-debugString":
                    print workingContext.debugString()
                else:  # "-best"
                    _store.dumpNested(workingContext, _outSink,
                            flags=option_flags[option_format])




############################################################ Main program
    
if __name__ == '__main__':
    import os
    doCommand()

