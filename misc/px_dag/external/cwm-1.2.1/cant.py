#! /usr/bin/python
"""CAnonicalize N-Triples

Options:

--verbose       -v      Print what you are doing as you go
--help          -h      Print this message and exit
--from=uri      -f uri  Specify an input file (or web resource)
--diff=uri      -d uri  Specify a difference file

Can have any number of --from <file> parameters, in which case files are
merged. If none are given, /dev/stdin is used.

If any diff files are given then the diff files are read merged separately
and compared with the input files. the result is a list of differences
instead of the canonicalizd graph. This is NOT a minimal diff.
Exits with nonzero system status if graphs do not match.
 
This is an independent n-triples cannonicalizer. It uses heuristics, and
will not terminate on all graphs. It is designed for testing:  the output and
the reference output are both canonicalized and compared.

It uses the very simple NTriples format. It is designed to be independent
of the SWAP code so that it can be used to test the SWAP code. It doesn't
boast any fancy algorithms - just tries to get the job done for the small
files in the test datasets.

The algorithm to generate a "signature" for each bnode. This is just found by looking in its immediate viscinity, treating any local bnode as a blank.
Bnodes which have signatures
unique within the graph can be allocated cannonical identifiers as a function
of the ordering of the signatures. These are then treated as fixed nodes.
If another pass is done of the new graph, the signatures are more distinct.

This works for well-labelled graphs, and graphs which don't have large areas
of interconnected bnodes or large duplicate areas. A particular failing
is complete lack of treatment of symmetry between bnodes.

References:
 .google graph isomorphism
 See also eg http://www.w3.org/2000/10/rdf-tests/rdfcore/utils/ntc/compare.cc
 NTriples: see http://www.w3.org/TR/rdf-testcases/#ntriples
 
 Not to mention,  published this month by coincidence:
  Kelly, Brian, [Whitehead Institute]  "Graph cannonicalization", Dr Dobb's Journal, May 2003.
 
 $Id: cant.py,v 1.15 2007/06/26 02:36:15 syosi Exp $
This is or was http://www.w3.org/2000/10/swap/cant.py
W3C open source licence <http://www.w3.org/Consortium/Legal/copyright-software.html>.

2004-02-31 Serious bug fixed.  This is a test program, shoul dbe itself tested.
                Quis custodiet ipsos custodes?
"""
# canticle - Canonicalizer of NTriples Independent of Cwm , Llyn, Etc. ?
import os
import sys
import urllib
try:
    from swap import uripath  # http://www.w3.org/2000/10/swap/
except ImportError:
    import uripath
from sys import stderr, exit
import uripath



import getopt
import re
import types

name = "[A-Za-z][A-Za-z0-9]*" #http://www.w3.org/TR/rdf-testcases/#ntriples
nodeID = '_:' + name
uriref = r'<[^>]*>'
language = r'[a-z0-9]+(?:-[a-z0-9]+)?'
string_pattern = r'".*"'   # We know in ntriples that there can only be one string on the line
langString = string_pattern + r'(?:@' + language + r')?'
datatypeString = langString + '(?:\^\^' + uriref + r')?' 
#literal = langString + "|" + datatypeString
object =  r'(' + nodeID + "|" + datatypeString + "|" + uriref + r')'
ws = r'[ \t]*'
com = ws + r'(#.*)?[\r\n]*' 
comment = re.compile("^"+com+"$")
statement = re.compile( ws + object + ws + object + ws + object  + com) # 


#"

def usage():
    print __doc__

def loadFiles(testFiles):
    graph = []
    WD = "file://" + os.getcwd() + "/"

    for fn in testFiles:
        if verbose: stderr.write("Loading data from %s\n" % fn)

        uri = uripath.join(WD, fn)
        inStream = urllib.urlopen(uri)
        while 1:
            line = inStream.readline()
            if line == "" : break           
#           if verbose: stderr.write("%s\n" % line)
            m = comment.match(line)
            if m != None: continue
            m = statement.match(line)
            if m == None:
                stderr.write("Syntax error: "+line+"\n")
                if verbose:
                    [stderr.write('%2x ' % ord(c)) for c in line]
                    stderr.write('\n')
                exit(-1)
            triple = m.group(1), m.group(2), m.group(3)
            if verbose: stderr.write( "Triple: %s  %s  %s.\n" % (triple[0], triple[1], triple[2]))
            graph.append(triple)
    if verbose: stderr.write("\nThere are %i statements in graph\n" % (len(graph)))
    return graph

def main():
    testFiles = []
    diffFiles = []
    global ploughOn # even if error
    ploughOn = 0
    global verbose
    verbose = 0
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:d:iv",
            ["help", "from=", "diff=", "to=", "ignoreErrors", "verbose"])
    except getopt.GetoptError:
        # print help information and exit:
        usage()
        sys.exit(2)
    output = None
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        if o in ("-v", "--verbose"):
            verbose = 1
        if o in ("-i", "--ignoreErrors"):
            ploughOn = 1
        if o in ("-f", "--from"):
            testFiles.append(a)
        if o in ("-d", "--diff", "--to"):
            diffFiles.append(a)

    

    if testFiles == []: testFiles = [ "/dev/stdin" ]
    graph = loadFiles(testFiles)
    graph = canonicalize(graph)
    
    if diffFiles != []:
        graph2 = loadFiles(diffFiles)
        graph2 = canonicalize(graph2)
        d = compareCanonicalGraphs(graph, graph2)
        if d != 0:
            sys.exit(d)
    else:
        serialize(graph)


def compareCanonicalGraphs(g1, g2):

    inserted, deleted = [], []
    g1.sort()
    g2.sort()
    i1, i2 = 0,0
    while 1:
        if i1 == len(g1):
            inserted = inserted + g2[i2:]
            if verbose: stderr.write("All other %i triples were inserted.\n" % (len(g2)-i2))
            break
        if i2 == len(g2):
            deleted = deleted + g1[i1:]
            if verbose: stderr.write("All other %i triples were deleted.\n" % (len(g1)-i1))
            break
        d = cmp(g1[i1], g2[i2]) # 1-2
        if d==0:
            if verbose: stderr.write("Common:   %s %s %s.\n" % g2[i2])
            i1 += 1
            i2 += 1
        elif d<0:
            if verbose: stderr.write("Deleted:  %s %s %s.\n" % g1[i1])
            deleted.append(g1[i1])
            i1 += 1
        else:
            if verbose: stderr.write("Inserted: %s %s %s.\n" % g2[i2])
            inserted.append(g2[i2])
            i2 += 1
    for triple in deleted:
        print "- %s %s %s." % triple
    for triple in inserted:
        print "+ %s %s %s." % triple
    number = len(deleted) + len(inserted)
    if verbose:
        if number == 0: stderr.write("FILES MATCH.\n")
        else: stderr.write("FILES DIFFER.  (%i statements by our count)\n"% number)
    return number

def canonicalize(g):
    "Do our best with this algo"
    dups, graph, c = canon(g)
    while dups != 0:
        newDups, graph, c = canon(graph, c)
        if newDups == dups:
            exit(-2) # give up
        dups = newDups
    return graph

def serialize(graph):
    graph.sort()
    if verbose: print "# Canonicalized:"
    for t in graph:
        for x in t:
            if x.startswith("__"): x = x[1:]
            print x,
        print "."

def compareFirst(a,b):
    "Compare consistently nested lists of strings"
    d = cmp(`a[0]`, `b[0]`)
    if verbose:
        if d<0: stderr.write("Comparing:  %s]\n  LESS THAN %s\n" % (`a`,`b`))
        elif d>0: stderr.write("Comparing:  %s]\n  LESS THAN %s\n" % (`b`,`a`))
        else: stderr.write("Comparing:  %s]\n     EQUALS %s\n" % (`b`,`a`))
    return d
    
    #@@@@@@@@@@@@
    if a==None and b == None: return 0
    if a == None: return -1
    if b == None: return 1
    if isinstance(a, types.IntType):
        if isinstance (b,types.IntType): return a-b
        else:
            return -1  # Ints are less than strings or lists
    if isinstance(a, types.StringTypes):
        if isinstance (b, types.IntType): return 1
        if isinstance (b,types.StringTypes):
            if a < b: return -1
            if a > b: return 1
            return 0
        else:
            return -1  # Strings are less than lists
    else:  # a is list
        if isinstance (b,types.StringTypes):
            return 1
        else: # list vs list
#           assert isinstance(a, types.ListType) or isinstance(a, TupleType)
            if len(a) < len(b): return -1
            if len(a) > len(b): return 1
            for i in range(len(a)):
                d = compare(a[i], b[i], level+1)
                if d != 0: return d
            return 0
                
def canon(graph, c0=0):
    """Try one pass at canonicalizing this using 1 step sigs.
    Return as a triple:
    - The new graph
    - The number of duplicate signatures in the bnodes
    - The index number for th enext constant to be generated."""
    nextBnode = 0
    bnodes = {}
    pattern = []
    signature = []
    canonical = {}
    for j in range(len(graph)):
        triple = graph[j]
        pat = []
        for i in range(3):
            if triple[i].startswith("_:"):
                b = bnodes.get(triple[i], None)
                if b == None:
                    b = nextBnode
                    nextBnode = nextBnode + 1
                    bnodes[triple[i]] = b
                    signature.append([])
                pat.append(None)
            else:
                pat.append(triple[i])
        pattern.append(pat)
        for i in range(3):
            if triple[i].startswith("_:"):
                b = bnodes[triple[i]]
                signature[b].append((i, pat))

    if verbose: stderr.write("\n")
    n = nextBnode
    s = []
    for i in range(n):
        signature[i].sort()   # Signature is now intrinsic to the local environment of that bnode.
        if verbose: stderr.write( "Bnode %3i) %s\n\n" % (i, signature[i]))
        s.append((signature[i], i))
    s.sort(compareFirst)
    
    dups = 0
    c = c0
    if verbose: stderr.write("\nIn order\n")
    for i in range(n):
        sig, original = s[i]
        if verbose: stderr.write("%3i) Orig: %i Sig:%s\n" %(i, original, sig))
        if i != n-1 and s[i][0] == s[i+1][0]:
            if verbose: stderr.write(
             "@@@ %3i]  %i and %i have same signature: \n\t%s\nand\t%s\n" % (
                    i, s[i][1], s[i+1][1], s[i][0], s[i+1][0]))
            dups = dups + 1
        elif i != 0 and s[i][0] == s[i-1][0]:
            if verbose: stderr.write( "@@@ %3i]  %i and %i have same signature: \n\t%s\nand\t%s\n" % (
                    i, s[i][1], s[i-1][1], s[i][0], s[i-1][0]))
        else:
            canonical[original] = c
            if verbose: stderr.write( "\tBnode#%i canonicalized to new fixed C%i\n" %(s[i][1], c))
            c = c + 1
            
    newGraph = []
    for j in range(len(graph)):
        triple = graph[j]
        newTriple = []
        for i in range(3):
            x = triple[i]
            if x.startswith("_:"):
                b = bnodes[x]
                c1 = canonical.get(b, None)
                if c1 != None:
                    x = "__:c" + str(c1) # New name
            newTriple.append(x)
        newGraph.append((newTriple[0], newTriple[1], newTriple[2]))
    if verbose: stderr.write("Iteration complete with %i duplicate signatures\n\n" %dups)
    return dups, newGraph, c



if __name__ == "__main__":
    main()



# ends
