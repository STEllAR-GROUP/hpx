#! /usr/bin/python
"""


$Id: cwm_list.py,v 1.15 2007/11/18 02:13:20 syosi Exp $

List and set built-ins for cwm
http://www.w3.org/2000/10/swap/cwm_list.py

See cwm.py and the os module in python

"""


from term import LightBuiltIn, RDFBuiltIn, Function, ReverseFunction, \
    MultipleFunction, MultipleReverseFunction, \
    CompoundTerm, N3Set, List, EmptyList, NonEmptyList

from set_importer import Set

from diag import verbosity, progress
import uripath

from RDFSink import List_NS, Logic_NS

ListOperationsNamespace = "http://www.w3.org/2000/10/swap/list#"

####################################################################
#
#                    List handling   B U I L T - I N s
#
#
#   Light Built-in classes


class BI_first(RDFBuiltIn, Function):
    def evalObj(self, subj, queue, bindings, proof, query):
        if not isinstance(subj, NonEmptyList): return None
        return subj.first

class BI_rest(RDFBuiltIn, Function):
    def evalObj(self, subj, queue, bindings, proof, query):
        if not isinstance(subj, NonEmptyList): return None
        return subj.rest

class BI_last(LightBuiltIn, Function):
    def evalObj(self, subj, queue, bindings, proof, query):
        if not isinstance(subj, NonEmptyList): return None
        x = subj
        while 1:
            last = x
            x = x.rest
            if isinstance(x, EmptyList): return last.first

##class BI_map(LightBuiltIn, Function):
##    def evalObj(self,subj, queue, bindings, proof, query):
##        print subj
##        store = self.store
##        genID = store.genId()
##        print genID
##        hash = genID.rfind("#")
##        print genID[hash+1:]
##        symbol = genID[:hash]
##        mapped = store.symbol(symbol)
##        class Map(LightBuiltIn, Function):
##            def evalObj(self, subj, queue, bindings, proof, query):
##                print 'hi'
##                return subj
##        
##        mapped.internFrag(genID[hash+1:], Map)
##        return store.symbol(genID)


class BI_in(LightBuiltIn, MultipleReverseFunction):
    """Is the subject in the object?
    Returnes a sequence of values."""
    def eval(self, subj, obj, queue, bindings, proof, query):
        if not isinstance(obj, CompoundTerm): return None
        return subj in obj
        

    def evalSubj(self, obj, queue, bindings, proof, query):
        if not isinstance(obj, NonEmptyList) and not isinstance(obj, N3Set): return None
        rea = None
        return [x for x in obj]  # [({subj:x}, rea) for x in obj]

class BI_member(LightBuiltIn, MultipleFunction):
    """Is the subject in the object?
    Returnes a sequence of values."""
    def eval(self, subj, obj, queue, bindings, proof, query):
        if not isinstance(subj, CompoundTerm): return None
        return obj in subj

    def evalObj(self,subj, queue, bindings, proof, query):
        if not isinstance(subj, NonEmptyList) and not isinstance(subj, N3Set): return None
        rea = None
        return [x for x in subj] # [({obj:x}, rea) for x in subj]



class BI_append(LightBuiltIn, Function):
    """Takes a list of lists, and appends them together.


    """
    def evalObj(self, subj, queue, bindings, proof, query):
        if not isinstance(subj, NonEmptyList): return None
        r = []
        for x in subj:
            if not isinstance(x, List): return None
            r.extend([a for a in x])
        return self.store.newList(r)

class BI_members(LightBuiltIn, Function):
    """Makes a set from a list

    """
    def evaluateObject(self, subj):
        return Set(subj)
    
#  Register the string built-ins with the store

def register(store):

#    Done explicitly in llyn
#    list = store.internURI(List_NS[:-1])
#    list.internFrag("first", BI_first)
#    list.internFrag("rest", BI_rest)

    ns = store.symbol(ListOperationsNamespace[:-1])
    ns.internFrag("in", BI_in)
    ns.internFrag("member", BI_member)
    ns.internFrag("last", BI_last)
    ns.internFrag("append", BI_append)
    ns.internFrag("members", BI_members)
##    ns.internFrag("map", BI_map)
# ends

