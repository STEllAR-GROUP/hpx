#! /bin/python
"""Regression test harness for new versions of cwm

python retrest.py   <options>  <inputURIs>
Options:

--testsFrom=uri -f uri  Take test definitions from these files (in RDF/XML or N3 format)
                        Or just by themselves at end of command line after options
--normal        -n      Do normal tests, checking output NOW DEFAULT - NOT NEEDED
--chatty        -c      Do tests with debug --chatty=100 (flag just check doesn't crash)
--proof         -p      Do tests generating and cheking a proof (if a test:CwmProofTest)
--start=13      -s 13   Skip the first 12 tests
--verbose       -v      Print what you are doing as you go
--ignoreErrors  -i      Print error message but plough on though more tests if errors found
                        (Summary error still raised when all tests ahve been tried)
--cwm=../cwm.py         Cwm command is ../cwm
--help          -h      Print this message and exit

You must specify some test definitions, and normal or proofs or both,
or nothing will happen.

Example:    python retest.py -n -f regression.n3

 $Id: retest.py,v 1.45 2007/11/18 02:01:57 syosi Exp $
This is or was http://www.w3.org/2000/10/swap/test/retest.py
W3C open source licence <http://www.w3.org/Consortium/Legal/copyright-software.html>.

"""
from os import system, popen3
import os
import sys
import urllib

# From PYTHONPATH equivalent to http://www.w3.org/2000/10

from swap import llyn
from swap.myStore import load, loadMany, Namespace
from swap.uripath import refTo, base
from swap import diag
from swap.diag import progress
from swap.term import AnonymousNode


rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
test = Namespace("http://www.w3.org/2000/10/swap/test.n3#")
n3test = Namespace("http://www.w3.org/2004/11/n3test#")
rdft = Namespace("http://www.w3.org/2000/10/rdf-tests/rdfcore/testSchema#")
triage = Namespace("http://www.w3.org/2000/10/swap/test/triage#")

sparql_manifest = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#")
sparql_query = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-query#")
dawg_test = Namespace("http://www.w3.org/2001/sw/DataAccess/tests/test-dawg#")

import getopt
import sys
import re


normal = 1
chatty = 0
proofs = 0
verbose = 0
no_action = 0

def localize(uri):
    """Get URI relative to where this lives"""
    from swap import uripath
    return uripath.refTo("http://www.w3.org/2000/10/swap/test/retest.py", uri)

def problem(str):
    global ploughOn
    global problems
    sys.stderr.write(str + "\n")
    problems.append(str)
    if not ploughOn:
        sys.exit(-1)

#       raise RuntimeError(str)

def usage():
    print __doc__

from subprocess import Popen, call, PIPE

def execute(cmd1, noStdErr=False):
    global verbose, no_action
    if verbose: print "    "+cmd1
    if no_action: return
    if noStdErr:
        stderr = file('/dev/null', 'w')
    else:
        stderr = None
    result = call(cmd1, shell=True, stderr=stderr)
    if result != 0:
        raise RuntimeError("Error %i executing %s" %(result, cmd1))

def diff(case, ref=None, prog="diff -Bbwu"):
    global verbose
    if ref == None:
        ref = "ref/%s" % case
    if diag.print_all_file_names:  #for use in listing all files used in testing
        a = open('testfilelist','a')
        a.write(ref)
        a.write('\n')
        a.close()
    diffcmd = """%s %s ,temp/%s >,diffs/%s""" %(prog, ref, case, case)
    if verbose: print "  ", diffcmd
    if no_action: result = 0
    else: result = system(diffcmd)
    if result < 0:
        raise problem("Comparison fails: result %i executing %s" %(result, diffcmd))
    if result > 0: print "Files differ, result=", result
    d = urllib.urlopen(",diffs/"+case)
    buf = d.read()
    if len(buf) > 0:
        if just_fix_it == 0:
            print "#  If this is OK,   cp ,temp/%s %s" %(case, ref)
            print "######### Differences from reference output:\n" + buf
            return 1
        else:
            os.system("cp ,temp/%s %s" %(case, ref))
            return 0
    return result

def rdfcompare3(case, ref=None):
    "Compare NTriples fieles using the cant.py"
    global verbose
    if ref == None:
        ref = "ref/%s" % case
    diffcmd = """python ../cant.py -d %s -f ,temp/%s >,diffs/%s""" %(ref, case, case)
    if verbose: print "  ", diffcmd
    result = system(diffcmd)
    if result < 0:
        raise problem("Comparison fails: result %i executing %s" %(result, diffcmd))
    if result > 0: print "Files differ, result=", result
    d = urllib.urlopen(",diffs/"+case)
    buf = d.read()
    if len(buf) > 0:
#       print "#  If this is OK,   cp ,temp/%s %s" %(case, ref)
        print "######### Differences from reference output:\n" + buf
        return 1
    return result

def rdfcompare2(case, ref1):
        """Comare ntriples files by canonicalizing and comparing text files"""
        cant = "python ../cant.py"
        ref = ",temp/%s.ref" % case
        execute("""cat %s | %s > %s""" % (ref1, cant, ref))
        return diff(case, ref)


def rdfcompare(case, ref=None):
    """   The jena.rdfcompare program writes its results to the standard output stream and sets
        its exit code to 0 if the models are equal, to 1 if they are not and
        to -1 if it encounters an error.</p>
    """
    global verbose
    if ref == None:
        ref = "ref/%s" % case
    diffcmd = """java jena.rdfcompare %s ,temp/%s N-TRIPLE N-TRIPLE  >,diffs/%s""" %(ref, case, case)
    if verbose: print "  ", diffcmd
    result = system(diffcmd)
    if result != 0:
        raise problem("Comparison fails: result %s executing %s" %(result, diffcmd))
    return result



def main():
    global verbose, proofs, chatty, normal, no_action
    start = 1
    cwm_command='../cwm.py'
    python_command='python -tt'
    global ploughOn # even if error
    ploughOn = 0
    global verbose
    verbose = 0
    global just_fix_it
    just_fix_it = 0
    if diag.print_all_file_names:
        a = file('testfilelist','w')
        a.write('')
        a.close()
    try:
        opts, testFiles = getopt.getopt(sys.argv[1:], "h?s:nNcipf:v",
            ["help", "start=", "testsFrom=", "no-action", "No-normal", "chatty",
                "ignoreErrors", "proofs", "verbose","overwrite","cwm="])
    except getopt.GetoptError:
        # print help information and exit:
        usage()
        sys.exit(2)
    output = None
    for o, a in opts:
        if o in ("-h", "-?", "--help"):
            usage()
            sys.exit()
        if o in ("-v", "--verbose"):
            verbose = 1
        if o in ("-i", "--ignoreErrors"):
            ploughOn = 1
        if o in ("-s", "--start"):
            start = int(a)
        if o in ("-f", "--testsFrom"):
            testFiles.append(a)
        if o in ("-n", "--no-action"):
            no_action = 1
        if o in ("-N", "--No-normal"):
            normal = 0
        if o in ("-c", "--chatty"):
            chatty = 1
        if o in ("-p", "--proofs"):
            proofs = 1
        if o in ("--overwrite",):
            just_fix_it = 1
        if o in ("--cwm", "--the_end"):
            cwm_command=a

    
    assert system("mkdir -p ,temp") == 0
    assert system("mkdir -p ,diffs") == 0
    if proofs: assert system("mkdir -p ,proofs") == 0
    
    tests=0
    passes=0
    global problems
    problems = []
    
    REFWD="http://example.com/swap/test"
    WD = base()[:-1] 
    
    #def basicTest(case, desc, args)

    if verbose: progress("Test files:", testFiles)
    
    kb = loadMany(testFiles, referer="")
    testData = []
    RDFTestData  = []
    RDFNegativeTestData = []
    perfData = []
    n3PositiveTestData = []
    n3NegativeTestData = []
    sparqlTestData = []
#    for fn in testFiles:
#       print "Loading tests from", fn
#       kb=load(fn)
    
    for t in kb.each(pred=rdf.type, obj=test.CwmTest):
        verboseDebug = kb.contains(subj=t, pred=rdf.type, obj=test.VerboseTest)
        u = t.uriref()
        ref = kb.the(t, test.referenceOutput)
        if ref == None:
            case = str(kb.the(t, test.shortFileName))
            refFile = "ref/%s" % case
        else:
            refFile = refTo(base(), ref.uriref())
            case  = ""
            for ch in refFile:
                if ch in "/#": case += "_"
                else: case += ch  # Make up test-unique temp filename
        description = str(kb.the(t, test.description))
        arguments = str(kb.the(t, test.arguments))
        environment = kb.the(t, test.environment)
        if environment == None: env=""
        else: env = str(environment) + " "
        testData.append((t, t.uriref(), case, refFile, description, env, arguments, verboseDebug))

    for t in kb.each(pred=rdf.type, obj=rdft.PositiveParserTest):

        x = t.uriref()
        y = x.find("/rdf-tests/")
        x = x[y+11:] # rest
        for i in range(len(x)):
            if x[i]in"/#": x = x[:i]+"_"+x[i+1:]
        case = "rdft_" + x + ".nt" # Hack - temp file name
        
        description = str(kb.the(t, rdft.description))
#           if description == None: description = case + " (no description)"
        inputDocument = kb.the(t, rdft.inputDocument).uriref()
        outputDocument = kb.the(t, rdft.outputDocument).uriref()
        status = kb.the(t, rdft.status).string
        good = 1
        if status != "APPROVED":
            if verbose: print "\tNot approved: "+ inputDocument[-40:]
            good = 0
        categories = kb.each(t, rdf.type)
        for cat in categories:
            if cat is triage.ReificationTest:
                if verbose: print "\tNot supported (reification): "+ inputDocument[-40:]
                good = 0
##            if cat is triage.ParseTypeLiteralTest:
##                if verbose: print "\tNot supported (Parse type literal): "+ inputDocument[-40:]
##                good = 0
        if good:
            RDFTestData.append((t.uriref(), case, description,  inputDocument, outputDocument))

    for t in kb.each(pred=rdf.type, obj=rdft.NegativeParserTest):

        x = t.uriref()
        y = x.find("/rdf-tests/")
        x = x[y+11:] # rest
        for i in range(len(x)):
            if x[i]in"/#": x = x[:i]+"_"+x[i+1:]
        case = "rdft_" + x + ".nt" # Hack - temp file name
        
        description = str(kb.the(t, rdft.description))
#           if description == None: description = case + " (no description)"
        inputDocument = kb.the(t, rdft.inputDocument).uriref()
        status = kb.the(t, rdft.status).string
        good = 1
        if status != "APPROVED":
            if verbose: print "\tNot approved: "+ inputDocument[-40:]
            good = 0
        categories = kb.each(t, rdf.type)
        for cat in categories:
            if cat is triage.knownError:
                if verbose: print "\tknown failure: "+ inputDocument[-40:]
                good = 0
            if cat is triage.ReificationTest:
                if verbose: print "\tNot supported (reification): "+ inputDocument[-40:]
                good = 0
        if good:
            RDFNegativeTestData.append((t.uriref(), case, description,  inputDocument))



    for t in kb.each(pred=rdf.type, obj=n3test.PositiveParserTest):
        u = t.uriref()
        hash = u.rfind("#")
        slash = u.rfind("/")
        assert hash >0 and slash > 0
        case = u[slash+1:hash] + "_" + u[hash+1:] + ".out" # Make up temp filename
        
        description = str(kb.the(t, n3test.description))
#           if description == None: description = case + " (no description)"
        inputDocument = kb.the(t, n3test.inputDocument).uriref()
        good = 1
        categories = kb.each(t, rdf.type)
        for cat in categories:
            if cat is triage.knownError:
                if verbose: print "\tknown failure: "+ inputDocument[-40:]
                good = 0
        if good:
            n3PositiveTestData.append((t.uriref(), case, description,  inputDocument))


    for t in kb.each(pred=rdf.type, obj=n3test.NegativeParserTest):
        u = t.uriref()
        hash = u.rfind("#")
        slash = u.rfind("/")
        assert hash >0 and slash > 0
        case = u[slash+1:hash] + "_" + u[hash+1:] + ".out" # Make up temp filename
        
        description = str(kb.the(t, n3test.description))
#           if description == None: description = case + " (no description)"
        inputDocument = kb.the(t, n3test.inputDocument).uriref()

        n3NegativeTestData.append((t.uriref(), case, description,  inputDocument))

    for tt in kb.each(pred=rdf.type, obj=sparql_manifest.Manifest):
        for t in kb.the(subj=tt, pred=sparql_manifest.entries):
            name = str(kb.the(subj=t, pred=sparql_manifest.name))
            query_node = kb.the(subj=t, pred=sparql_manifest.action)
            if isinstance(query_node, AnonymousNode):
                data = ''
                for data_node in kb.each(subj=query_node, pred=sparql_query.data):
                    data = data + ' ' + data_node.uriref()

                inputDocument = kb.the(subj=query_node, pred=sparql_query.query).uriref()
            else:
                data = ''
                inputDocument = query_node.uriref()
            j = inputDocument.rfind('/')
            case = inputDocument[j+1:]
            outputDocument = kb.the(subj=t, pred=sparql_manifest.result)
            if outputDocument:
                outputDocument = outputDocument.uriref()
            else:
                outputDocument = None
            good = 1
            status = kb.the(subj=t, pred=dawg_test.approval)
            if status != dawg_test.Approved:
                print status, name
                if verbose: print "\tNot approved: "+ inputDocument[-40:]
                good = 0
            if good:
                sparqlTestData.append((tt.uriref(), case, name, inputDocument, data, outputDocument))
        



    for t in kb.each(pred=rdf.type, obj=test.PerformanceTest):
        x = t.uriref()
        theTime = kb.the(subj=t, pred=test.pyStones)
        description = str(kb.the(t, test.description))
        arguments = str(kb.the(t, test.arguments))
        environment = kb.the(t, test.environment)
        if environment == None: env=""
        else: env = str(environment) + " "
        perfData.append((x, theTime, description, env, arguments))

    testData.sort()
    cwmTests = len(testData)
    if verbose: print "Cwm tests: %i" % cwmTests
    RDFTestData.sort()
    RDFNegativeTestData.sort()
    rdfTests = len(RDFTestData)
    rdfNegativeTests = len(RDFNegativeTestData)
    perfData.sort()
    perfTests = len(perfData)
    n3PositiveTestData.sort()
    n3PositiveTests = len(n3PositiveTestData)
    n3NegativeTestData.sort()
    n3NegativeTests = len(n3NegativeTestData)
    sparqlTestData.sort()
    sparqlTests = len(sparqlTestData)
    totalTests = cwmTests + rdfTests + rdfNegativeTests + sparqlTests \
                 + perfTests + n3PositiveTests + n3NegativeTests
    if verbose: print "RDF parser tests: %i" % rdfTests

    for t, u, case, refFile, description, env, arguments, verboseDebug in testData:
        tests = tests + 1
        if tests < start: continue
        
        urel = refTo(base(), u)
    
        print "%3i/%i %-30s  %s" %(tests, totalTests, urel, description)
    #    print "      %scwm %s   giving %s" %(arguments, case)
        assert case and description and arguments
        cleanup = """sed -e 's/\$[I]d.*\$//g' -e "s;%s;%s;g" -e '/@prefix run/d' -e 's;%s;%s;g'""" % (WD, REFWD,
                                                                                                      cwm_command, '../cwm.py')
        
        if normal:
            execute("""CWM_RUN_NS="run#" %s %s %s --quiet %s | %s > ,temp/%s""" %
                (env, python_command, cwm_command, arguments, cleanup , case))  
            if diff(case, refFile):
                problem("######### from normal case %s: %scwm %s" %( case, env, arguments))
                continue

        if chatty and not verboseDebug:
            execute("""%s %s %s --chatty=100  %s  &> /dev/null""" %
                (env, python_command, cwm_command, arguments), noStdErr=True)   

        if proofs and kb.contains(subj=t, pred=rdf.type, obj=test.CwmProofTest):
            execute("""%s %s %s --quiet %s --base=a --why  > ,proofs/%s""" %
                (env, python_command, cwm_command, arguments, case))
            execute("""%s ../check.py < ,proofs/%s | %s > ,temp/%s""" %
                (python_command, case, cleanup , case)) 
            if diff(case, refFile):
                problem("######### from proof case %s: %scwm %s" %( case, env, arguments))
#       else:
#           progress("No proof for "+`t`+ " "+`proofs`)
#           progress("@@ %s" %(kb.each(t,rdf.type)))
        passes = passes + 1


    for u, case, name, inputDocument, data, outputDocument in sparqlTestData:
        tests += 1
        if tests < start: continue

        urel = refTo(base(), u)
        print "%3i/%i %-30s  %s" %(tests, totalTests, urel, name)
        inNtriples = case + '_1'
        outNtriples = case + '_2'
        try:
            execute("""%s %s %s --sparql=%s --filter=%s --filter=%s --ntriples > ',temp/%s'""" %
                    (python_command, cwm_command, data, inputDocument,
                     'sparql/filter1.n3', 'sparql/filter2.n3', inNtriples))
        except NotImplementedError:
            pass
        except:
            problem(str(sys.exc_info()[1]))
        if outputDocument:
            execute("""%s %s %s --ntriples > ',temp/%s'""" %
                    (python_command, cwm_command, outputDocument, outNtriples))
            if rdfcompare3(inNtriples, ',temp/' + outNtriples):
                problem('We have a problem with %s on %s' %  (inputDocument, data))


        passes += 1
        
    for u, case, description,  inputDocument, outputDocument in RDFTestData:
        tests = tests + 1
        if tests < start: continue
    
    
        print "%3i/%i)  %s   %s" %(tests, totalTests, case, description)
    #    print "      %scwm %s   giving %s" %(inputDocument, case)
        assert case and description and inputDocument and outputDocument
#       cleanup = """sed -e 's/\$[I]d.*\$//g' -e "s;%s;%s;g" -e '/@prefix run/d' -e '/^#/d' -e '/^ *$/d'""" % (
#                       WD, REFWD)
        execute("""%s %s --quiet --rdf=RT %s --ntriples  > ,temp/%s""" %
            (python_command, cwm_command, inputDocument, case))
        if rdfcompare3(case, localize(outputDocument)):
            problem("  from positive parser test %s running\n\tcwm %s\n" %( case,  inputDocument))

        passes = passes + 1

    for u, case, description,  inputDocument in RDFNegativeTestData:
        tests = tests + 1
        if tests < start: continue
    
    
        print "%3i/%i)  %s   %s" %(tests, totalTests, case, description)
    #    print "      %scwm %s   giving %s" %(inputDocument, case)
        assert case and description and inputDocument
#       cleanup = """sed -e 's/\$[I]d.*\$//g' -e "s;%s;%s;g" -e '/@prefix run/d' -e '/^#/d' -e '/^ *$/d'""" % (
#                       WD, REFWD)
        try:
            execute("""%s %s --quiet --rdf=RT %s --ntriples  > ,temp/%s 2>/dev/null""" %
            (python_command, cwm_command, inputDocument, case))
        except:
            pass
        else:
            problem("""I didn't get a parse error running python %s --quiet --rdf=RT %s --ntriples  > ,temp/%s
from test ^=%s
I should have.
""" %
            (cwm_command, inputDocument, case, u))

        passes = passes + 1


    for u, case, description, inputDocument in n3PositiveTestData:
        tests = tests + 1
        if tests < start: continue
    
    
        print "%3i/%i)  %s   %s" %(tests, totalTests, case, description)
    #    print "      %scwm %s   giving %s" %(inputDocument, case)
        assert case and description and inputDocument
#       cleanup = """sed -e 's/\$[I]d.*\$//g' -e "s;%s;%s;g" -e '/@prefix run/d' -e '/^#/d' -e '/^ *$/d'""" % (
#                       WD, REFWD)
        try:
            execute("""%s %s --grammar=../grammar/n3-selectors.n3  --as=http://www.w3.org/2000/10/swap/grammar/n3#document --parse=%s  > ,temp/%s 2>/dev/null""" %
            (python_command, '../grammar/predictiveParser.py', inputDocument, case))
        except RuntimeError:
            problem("""Error running ``python %s --grammar=../grammar/n3-selectors.n3  --as=http://www.w3.org/2000/10/swap/grammar/n3#document --parse=%s  > ,temp/%s 2>/dev/null''""" %
            ('../grammar/predictiveParser.py', inputDocument, case))
        passes = passes + 1

    for u, case, description, inputDocument in n3NegativeTestData:
        tests = tests + 1
        if tests < start: continue
    
    
        print "%3i/%i)  %s   %s" %(tests, totalTests, case, description)
    #    print "      %scwm %s   giving %s" %(inputDocument, case)
        assert case and description and inputDocument
#       cleanup = """sed -e 's/\$[I]d.*\$//g' -e "s;%s;%s;g" -e '/@prefix run/d' -e '/^#/d' -e '/^ *$/d'""" % (
#                       WD, REFWD)
        try:
            execute("""%s %s ../grammar/n3-selectors.n3  http://www.w3.org/2000/10/swap/grammar/n3#document %s  > ,temp/%s 2>/dev/null""" %
                (python_command, '../grammar/predictiveParser.py', inputDocument, case))
        except:
            pass
        else:
            problem("""There was no error executing ``python %s --grammar=../grammar/n3-selectors.n3  --as=http://www.w3.org/2000/10/swap/grammar/n3#document --parse=%s    > ,temp/%s''
            There should have been one.""" %
                ('../grammar/predictiveParser.py', inputDocument, case))

        passes = passes + 1


    timeMatcher = re.compile(r'\t([0-9]+)m([0-9]+)\.([0-9]+)s')
##    from test.pystone import pystones
##    pyStoneTime = pystones()[1]
    for u, theTime, description, env, arguments in perfData:
        tests = tests + 1
        if tests < start: continue
        
        urel = refTo(base(), u)
    
        print "%3i/%i %-30s  %s" %(tests, totalTests, urel, description)
        tt = os.times()[-1]
        a = system("""%s %s %s --quiet %s >,time.out""" %
                       (env, python_command, cwm_command, arguments))
        userTime = os.times()[-1] - tt
        print """%spython %s --quiet %s 2>,time.out""" % \
                       (env, cwm_command, arguments)
##        c = file(',time.out', 'r')
##        timeOutput = c.read()
##        c.close()
##        timeList = [timeMatcher.search(b).groups() for b in timeOutput.split('\n') if timeMatcher.search(b) is not None]
##        print timeList
##        userTimeStr = timeList[1]
##        userTime = int(userTimeStr[0])*60 + float(userTimeStr[1] + '.' + userTimeStr[2])
        pyCount = pyStoneTime * userTime
        print pyCount
        
    if problems != []:
        sys.stderr.write("\nProblems:\n")
        for s in problems:
            sys.stderr.write("  " + s + "\n")
        raise RuntimeError("Total %i errors in %i tests." % (len(problems), tests))

if __name__ == "__main__":
    main()


# ends
