#! /usr/bin/env python
"""


$Id: cwm_set.py,v 1.4 2007/06/26 02:36:15 syosi Exp $

set built-ins for cwm
http://www.w3.org/2000/10/swap/cwm_list.py

See cwm.py and the os module in python

"""


from term import LightBuiltIn, Function, ReverseFunction, MultipleFunction,\
    MultipleReverseFunction, \
    CompoundTerm, N3Set, List, EmptyList, NonEmptyList

from set_importer import Set

from diag import verbosity, progress
import uripath

from RDFSink import List_NS, Logic_NS

SetOperationsNamespace = "http://www.w3.org/2000/10/swap/set#"

###############################################################################################
#
#                    List handling   B U I L T - I N s
#
#
#   Light Built-in classes



class BI_in(LightBuiltIn, MultipleReverseFunction):
    """Is the subject in the object?
    Returnes a sequence of values."""
    def eval(self, subj, obj, queue, bindings, proof, query):
        if not isinstance(obj, CompoundTerm): return None
        return subj in obj
        

    def evalSubj(self, obj, queue, bindings, proof, query):
        if not isinstance(obj, NonEmptyList) and not isinstance(obj, N3Set): return None
        rea = None
        return [x or x in obj]  # [({subj:x}, rea) for x in obj]

class BI_member(LightBuiltIn, MultipleFunction):
    """Is the subject in the object?
    Returnes a sequence of values."""
    def eval(self, subj, obj, queue, bindings, proof, query):
        if not isinstance(subj, CompoundTerm): return None
        return obj in subj

    def evalObj(self,subj, queue, bindings, proof, query):
        if not isinstance(subj, NonEmptyList) and not isinstance(subj, N3Set): return None
        rea = None
        return subj # [({obj:x}, rea) for x in subj]


class BI_union(LightBuiltIn, Function):
    """Takes a set or list of sets, and finds the union

    """
    def evaluateObject(self, subj):
        ret = Set()
        for m in subj:
            ret.update(m)
        return ret

class BI_intersection(LightBuiltIn, Function):
    """Takes a set or list of sets, and finds the intersection


    """
    def evaluateObject(self, subj):
        ret = None
        for m in subj:
            if ret is None:
                ret = Set(m)
            else:
                ret.intersection_update(m)
        if ret is None:
            return Set()
        return ret

class BI_symmetricDifference(LightBuiltIn, Function):
    """Takes a set or list of two sets, and finds the symmetric difference


    """
    def evaluateObject(self, subj):
        if len(subj) != 2:
            raise ValueError('A symmetric difference of more than two things makes no sense')
        ret = Set()
        for m in subj:
            ret.symmetric_difference_update(m)
        return ret

class BI_difference(LightBuiltIn, Function):
    """Takes a list of two sets, and finds the difference


    """
    def evaluateObject(self, subj):
        if len(subj) != 2:
            raise ValueError('A symmetric difference of more than two things makes no sense')
        difference = N3Set.difference
        return difference(subj[0], subj[1])

class BI_oneOf(LightBuiltIn, ReverseFunction):
    """ Make a set from a list

    """
    def evaluateSubject(self, obj):
        return Set(obj)

#####################################
##
##   Sets and Formulae --- set:in is the inverse of most of these
##
class BI_subjects(LightBuiltIn, Function):
    """Return the set of subjects used in a formula

    """
    def evalObj(self, subj, queue, bindings, proof, query):
        if not isinstance(subj, Formula):
            raise ValueError('Only a formula has statements')
        return N3Set([x.subject() for x in subj])

class BI_predicates(LightBuiltIn, Function):
    """Return the set of subjects used in a formula

    """
    def evalObj(self, subj, queue, bindings, proof, query):
        if not isinstance(subj, Formula):
            raise ValueError('Only a formula has statements')
        return N3Set([x.predicate() for x in subj])

class BI_objects(LightBuiltIn, Function):
    """Return the set of subjects used in a formula

    """
    def evalObj(self, subj, queue, bindings, proof, query):
        if not isinstance(subj, Formula):
            raise ValueError('Only a formula has statements')
        return N3Set([x.object() for x in subj])

class BI_triples(LightBuiltIn, Function):
    """Return the set of triple used in a formula

    """
    def evalObj(self, subj, queue, bindings, proof, query):
        if not isinstance(subj, Formula):
            raise ValueError('Only a formula has statements')
        return N3Set([x.asFormula() for x in subj])

##class BI_existentials(LightBuiltIn, Function):
##    """Return the set of subjects used in a formula
##
##    """
##    def evalObj(self, subj, queue, bindings, proof, query):
##        if not isinstance(subj, Formula):
##            raise ValueError('Only a formula has statements')
##        return N3Set(subj.existentials())
##
##class BI_universals(LightBuiltIn, Function):
##    """Return the set of subjects used in a formula
##
##    """
##    def evalObj(self, subj, queue, bindings, proof, query):
##        if not isinstance(subj, Formula):
##            raise ValueError('Only a formula has statements')
##        return N3Set(subj.universals())

#  Register the string built-ins with the store

def register(store):

#    Done explicitly in llyn
#    list = store.internURI(List_NS[:-1])
#    list.internFrag("first", BI_first)
#    list.internFrag("rest", BI_rest)

    ns = store.symbol(SetOperationsNamespace[:-1])
    ns.internFrag("in", BI_in)
    ns.internFrag("member", BI_member)
    ns.internFrag("union", BI_union)
    ns.internFrag("intersection", BI_intersection)
    ns.internFrag("symmetricDifference", BI_symmetricDifference)
    ns.internFrag("difference", BI_difference)
    ns.internFrag("oneOf", BI_oneOf)
# ends

