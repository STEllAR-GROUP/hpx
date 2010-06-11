from terms import URI, Variable, Fact, Exivar

from helpers import removedups,  getOccurences, keysToList, sortVars, Memory, convertBNodeToFact, isListVarNode, INCLUDES
from prooftrace import ProofTrace
from sets import Set
from helpers import FIRST, REST, TYPE, INCLUDES, LOG_SEMANTICS as SEMANTICS, IMPLIES, NIL
import time
import copy
import exception

#from pychinko.N3Loader import formulas

py_lists = dict()

#add initial facts that match the lists at the very beginning - shouldnt screw up anything

class AlphaNode:
    def __init__(self, pattern):        
        self.pattern = pattern
        #print "pattern", pattern
        self.ind = Memory()     ###ind=memory, or ind=index
        self.betaNodes = list()
        self.vars = [v for v in pattern if isinstance(v, Variable)]
        sortVars(self.vars)
        self.svars = list()
        self.dependents = list()
        self.dependsOn = list()

            
    def clear(self):
        self.ind = Memory()

    #####Make this a Pattern method -- multiple arity patterns
    def getvar(self, var, fact):
        """I return the value of var in fact according to our pattern."""
        if var in self.pattern:
            pos = self.pattern.index(var)
            return fact[pos]
        else:
            raise exception.UnboundRuleVariable(var, self.pattern)

    def getbindings(self, row, useBuiltin=None):
        """I return a set of bindings for row, where row is a set of values from
        my AlphaMemory (note that rows that do not have constants, only variable values.)"""
        bindings = dict()
        key = removedups(self.svars + self.vars)
        for i, val in enumerate(key):
            bindings[val] = row[i]
        #print "key alpha:",key
        #print "row alpha:",row

        return bindings

    def  getkey(self):
        return removedups(self.svars + self.vars)
    
    def getrest(self, fact, sharedVars):
        """I return the unshared variable values for a given fact 
        according to our pattern. Note that right now we need to 
        pass the *shared* variable list as a parameter, but in our 
        Rete network we would just make that an attribute of every 
        ANode in compilation."""
        vals = list()
        for i, v in enumerate(self.pattern):
            if isinstance(v, Variable) and v not in sharedVars:
                vals.append(fact[i])
        return vals

    def setupIndex(self):
        if not self.svars:
            self.svars = [self.vars[0]] # need to account for 0 var case
        # len(shared) <= len(self.vars)
        # We need to remove dups.
        # unshared and shared are *disjoint*, so only need to remove
        # dups in each
        self.unshared = list(removedups([v for v in self.vars if v not in self.svars]))

    ####Make this a Pattern method
    def match(self, fact):
        """Determines whether the fact matches the node's pattern"""
        bindings = dict()
        for p, f in zip(self.pattern, fact):
            if not isinstance(p, Variable):
                if p != f:
                    return False
            elif p not in bindings:
                bindings[p] = f
            elif bindings[p] != f:
                return False
        return bindings                

    def addAll(self, facts):
        for f in facts:
            self.add(f)
            
    def add(self, fact):
        bindings = self.match(fact)        
        if bindings:
            #make sure key has shared vars first
            key = removedups(self.svars + self.vars)
            return self.index(self.ind, bindings, key, fact)
        else:
            return False
        
    def getJustification(fact):
        """I take a fact and return its justification set (another set of facts)."""
        if fact.s in self.ind:
            if fact.p in self.ind[fact.s]:
                val = self.ind[fact.s][fact.p]
                if fact.o in val[fact.o]:
                    return val[fact.o]
        return None

    def exists(self, fact, bindings):
        """Check the index for the presence of the fact
        (as expressed as a set of bindings returned by match)"""
        key = self.svars + self.unshared
        return self.__exists(bindings, self.ind, key)

    def __exists(self, bindings, ind, key):
        # vars are in reverse shared/unshared sorted
        if key: # Still comparison work to be done
            cur = key.pop(0)
            if bindings[cur] in ind:
                return self.existshelp(bindings, ind[bindings[cur]], key)
            else:
                return False
        else: # We succeded in getting to the bottom of the index
            return True

    def clear(self):
        """I clear the memory of this AlphaNode.  This is only called
        from unit tests."""
        self.ind = Memory()
        
    def index(self, ind, bindings, key, factAdded=None):
        if key: # Still work to be done
            cur = bindings[key.pop(0)]  #pop(0) pops the first item off
            if cur not in ind:
                # So we know the fact doesn't exist
                if key:
                    ind[cur] = Memory()  # just a dictionary -- intended for sorted join 
                    return self.index(ind[cur], bindings, key, factAdded)
                else: # At the bottom, and the fact still doesn't exist
                    # Create justification set, used for proof tracing, and stick it
                    # as the inner most value in the memory
                    pt = ProofTrace()
                    pt.addPremise(factAdded)
                    ind[cur] = tuple(pt)   
                    return True #it was added
            else:
                if key: # Perhaps the fact does exist
                    return self.index(ind[cur], bindings, key, factAdded)
                else:
                    # It definitely exists.
                    return False

    def __repr__(self):
        return """AlphaNode(%s)(Mem: %s)""" %(str(self.pattern), str(self.ind))

class BetaNode:
    """A Beta node is a *join* of two other nodes.  The other nodes
    maybe any mix of alphas and betas.  An alpha may be joined against
    itself.  The Beta has a set of *shared variables*, that is,
    variables that appear in both nodes (there may be none)."""
    def __init__(self, lnode, rnode):
        from builtins import builtinp
        self.lnode = lnode
        self.rnode = rnode
        #####Make this opaque
        if isinstance(lnode, BetaNode):
            self.lvars = lnode.pattern
        elif isinstance(lnode, AlphaNode):
            self.lvars = lnode.vars
        self.rvars = rnode.vars
        # Detect builtins
        if isinstance(rnode, AlphaNode):
            if builtinp(self.rnode):
                self.svars = lnode.svars
            else:
                # store svars in lexical order
                self.svars = [v for v in self.lvars if v in self.rvars]
        else:
            self.svars = [v for v in self.lvars if v in self.rvars]
        sortVars(self.svars) #destructively sort vars
        self.parents = list()
        self.children = list()
        self.pattern = removedups(self.svars + self.lvars + self.rvars)
        
        # Redundant var, will be refactored out
        self.vars = self.pattern
        self.builtinInput = None
        self.ind = Memory()
        self.inferredFacts = Set()
        # a pointer to the rule node (which contains the rhs)
        self.rule = None

    ####This is 'match' and hence should be a pattern method
    def getbindings(self, row, useBuiltin=None):
        bindings = dict()
        #check for builtins
        if useBuiltin:            
            key = useBuiltin.indexKey()
        else:
            if isinstance(self.lnode, AlphaNode):
                key = removedups(self.lnode.svars + self.lnode.vars + self.rnode.vars)
            elif isinstance(self.lnode, BetaNode):
                key = removedups(self.lnode.pattern + self.rnode.vars)        
        for i, v in enumerate(key):
            bindings[v] = row[i]
        return bindings

    def  getkey(self):        
        if isinstance(self.lnode, AlphaNode):
            key = removedups(self.lnode.svars + self.lnode.vars + self.rnode.vars)
        elif isinstance(self.lnode, BetaNode):
            key = removedups(self.lnode.pattern + self.rnode.vars)
        return key
    
    def add(self, row, justification=None):
        # store results in regular order
        key = copy.copy(self.pattern)
        bindings = self.getbindings(row)
        if bindings:
            return self.index(self.ind, row, bindings, key, justification)
        return False

    def index(self, ind, row, bindings, key, justification=None):
        if key: # Still work to be done
            cur = bindings[key.pop(0)]  #pop(0) pops the first item off
            if cur not in ind:
                # So we know the fact doesn't exist
                if key:
                    ind[cur] = Memory()
                    return self.index(ind[cur], row, bindings, key, justification)
                else: # At the bottom, and the factd still doesn't exist
                    if justification:
                        ind[cur] = tuple(justification)
                    else:
                        ind[cur] = tuple()
                    return True #it was added
            else:
                if key: # Perhaps the fact does exist
                    return self.index(ind[cur], row, bindings, key, justification)
                else:
                    # It definitely exists.
                    return False
 
    def hasBuiltins(self):
        from builtins import builtinp, funcBuiltinp
        """Return True if I have builtins connected to me, otherwise False.
        Also set a pointer to the builtins"""
        # If left and right are builtins, they will have the same input node (does this hold for  func. builtins, too?)        
        if builtinp(self.lnode):            
            self.builtinInput = self.lnode.getInputNode()            
        if builtinp(self.rnode):            
            self.builtinInput = self.rnode.getInputNode()        
        return bool(self.builtinInput)


    def bindPossibleVar(self, possVar, bindings):        
        if isinstance(possVar, Variable):            
            return bindings[possVar]
        else:
            return possVar


    
    def evalBuiltins(self, returnBindings=False):
        t = time.time()
        """I evaluate my attached builtins nodes."""
        from builtins import builtinp, funcBuiltinp                        
        #Three cases to handle: left is a builtin and right is a plain anode,
        #left is a plain anode and right is a builtin, or both the left
        #and right are builtins
        evaldRows = list()
        builtinNodes = list()
        
        
        if builtinp(self.lnode):
            builtinNodes.append(self.lnode)            
        if builtinp(self.rnode):
            builtinNodes.append(self.rnode)

        
        for bi in builtinNodes:
            
            inputNode = bi.getInputNode()            
            builtinInput = keysToList(inputNode.ind)
          
            
            for row in builtinInput:
                
                row = row[0] #Needed since row[1] is the justification set

                #print row
                #print bi.pattern
                bindings = inputNode.getbindings(row, useBuiltin=bi)                
                #need to check and substitute bi.pattern.s and bi.pattern.o for possible bindings here                
                #check and substitute if subj or obj are lists, they also might have  variables in the lists                
                
                subjInput = convertBNodeToFact(bi.pattern.s)
                                
                objInput = convertBNodeToFact(bi.pattern.o)
                
                
                if isinstance(bi.pattern.s, Variable):                    
                    if bi.pattern.s in bindings:                    
                        subjInput = convertBNodeToFact(bindings[bi.pattern.s])
                        
                if isinstance(bi.pattern.o, Variable):                    
                    if bi.pattern.o  in bindings:
                        objInput = convertBNodeToFact(bindings[bi.pattern.o])

                if subjInput in py_lists:
                    subjInput = [self.bindPossibleVar(x, bindings) for x in py_lists[subjInput]]

                if objInput in py_lists:
                    objInput = [self.bindPossibleVar(x, bindings) for x in py_lists[objInput]]
                
                
                # print "builtin", bi, subjInput, objInput                    
                result = bi.evaluate(subjInput, objInput)
                
                
                
                if result:

                    #hack: append the bindings if it's log:includes
                    if bi.pattern.p == INCLUDES:                    
                        bindings[bi.pattern.s]= subjInput
                        bindings[bi.pattern.o]= objInput
                        row.append(subjInput)
                        row.append(objInput)                        

                    
                    #a bit inefficient
                    if returnBindings:
                        evaldRows.append(bindings)
                    else:
                        evaldRows.append([row, []])  #Empty justification for now
                    #add to memory --
                    ##NOTE: Need to add justification (proof tracing) for data resulting
                    ##from builtins
                    #print "row after bindings", row
                    #if it's a functional builtin, it's different - need to return result and store in memory
                    
                    if funcBuiltinp(self.lnode) or funcBuiltinp(self.rnode):
                        #have no idea why it works
                        row.append(result)                        
                        #print "row added in evalbuiltins:", row
                        #add this fact
                        self.add(row)
                        #print "___row:", row
                    else:
                        #print "___row:", row
                        self.add(row)
                    
        #print "evaldRows:", evaldRows
        print "evalBultins time:", time.time() - t                        
        return evaldRows

    def getSharedVars(self, node1, node2):
        if isinstance(node1, BetaNode):
            lvars = node1.pattern
        else:
            lvars = node1.vars
        if isinstance(node2, BetaNode):
            rvars = node2.pattern
        else:
            rvars = node2.vars        
        result = removedups([item for item in lvars \
                             if item in rvars])
        return result
                             
    #optimize join: put node with smaller memory to the left
    #question: how can we switch the nodes in the join in the most painless way possible?
    #goal is not to affect other code at all, do all of the stuff in the join/joinhelper fcn.
    
    def join(self, useBindings=False):

        """
        print "\n\n"
        print "self key:", self.getkey()
        print "join left:", self.lnode.pattern
        #for row in keysToList(self.lnode.ind):
        #    print "left row",row[0]        
            
        print "memory size of left node:", len(keysToList(self.lnode.ind))
        print "left key:", self.lnode.getkey()

        
        print ""
        print "join right:", self.rnode.pattern
       # for row in keysToList(self.rnode.ind):
        #    print "right row",row[0]
        print "memory size of right node:", len(keysToList(self.rnode.ind))
        print "right key:", self.rnode.getkey()
        
        
        switched = False
        t = time.time()        
        if len(keysToList(self.rnode.ind)) < len(keysToList(self.lnode.ind)):
            print "switched"
            switched = True

        print "keysToList time:", time.time() - t        """
                
        
        from builtins import builtinp                    
        if self.lnode == self.rnode:        
            from builtins import builtinp, funcBuiltinp        
            if builtinp(self.lnode):            
                self.builtinInput = self.lnode.getInputNode()
                #problem is, here builtin input is empty
                return self.evalBuiltins(returnBindings=useBindings)                  
            else:
                self.pattern = self.lnode.vars
                if self.lnode.ind:
                    return keysToList(self.lnode.ind)
                else:
                    return []        
        elif self.hasBuiltins():            
            return self.evalBuiltins(returnBindings=useBindings)
        else:
            #right node is always an alpha node,
            #left one is an alphanode in the first join, beta node aftewards

            #compare the memory of both nodes, we want the smaller one to the left
            #
            
            if isListVarNode(self.rnode):                
                
                rows = keysToList(self.lnode.ind)
                
                for row in rows:
                    row = row[0]
                    #print "rows:", row
                    #print "left node key:", self.lnode.getkey()
                    #left node key does not correspond to the bindings
                    
                    #bindings =  self.lnode.getbindings(row)
                    #get the bindings according to its own key, above line didnt work
                    bindings = dict()
                    key = self.getkey()
                    try:
                        for i, v in enumerate(key):
                            bindings[v] = row[i]
                    except:
                        print "bindings=%s" % bindings
                        print "row=%s" % row
                        print "v=%s" % v
                        print "i=%s" % i
                        raise

                    #copy the values for the variable in pattern.o
                    if self.rnode.pattern.o in bindings:                    
                        #copy it back to the original node - easiest way is to add it as a fact
                        #also add bindings for the bnodes in subj
                        for var in self.rnode.vars:
                            if isinstance(var, Exivar):                                
                                self.rnode.add(Fact(convertBNodeToFact(var), self.rnode.pattern.p, bindings[self.rnode.pattern.o]))
                                #print "fact added:",convertBNodeToFact(var), self.rnode.pattern.p, bindings[self.rnode.pattern.o]
                    
                    elif isinstance(self.rnode.pattern.s, Exivar):
                        print "something wrong happened - %s was supposed to be bound", self.rnode.pattern.s
                        
                               
            if isinstance(self.lnode, AlphaNode):
                key = self.lnode.svars                                
            elif isinstance(self.lnode, BetaNode):
                
                if not Set(self.lnode.vars).intersection(Set(self.rnode.vars)):
                    key = list()
                else:
                    #regenerate key here? was using self.svars before but was buggy
                    key = self.getSharedVars(self.lnode, self.rnode)
                    
                    
            joinResults = list()
            if not key:
                #no shared variables and no matched values, so store and return the union
                #of the two memories
                """if switched:
                    leftMemory = keysToList(self.rnode.ind)
                    rightMemory = keysToList(self.lnode.ind)
                else:"""
                leftMemory = keysToList(self.lnode.ind)
                rightMemory = keysToList(self.rnode.ind)
                    
                for lm, ljust in leftMemory:
                    for rm, rjust in rightMemory:                        
                        """if switched:
                            row =  rm + lm
                        else:"""
                        row = lm + rm
                        joinResults.append([row, []])
                        ####NOTE -- Need to add justification (proof tracing) for this case
                        self.add(row, [])
            else:
                t = time.time()
                """if switched:
                    joinResults = self.joinhelper(self.rnode.ind, self.lnode.ind, key, [], [], switched)
                else:"""
                joinResults = self.joinhelper(self.lnode.ind, self.rnode.ind, key, [], [])
                print "joinhelper time:", time.time() - t        
            
            return joinResults
        

    def joinhelper(self, leftMemory, rightMemory, sharedVars, matchedValues, results, switched=False):
        
        #all shared variable values matched        
        if sharedVars:
            bindings = dict()
            #do this until we run out of shared variables to compare
            for lmk in leftMemory:
                #print "lmk:", lmk
                if lmk in rightMemory:
                    #match on shared variable, continue to next level
                    #leftMemory and rightMemory are nested dictionaries
                    self.joinhelper(leftMemory[lmk], rightMemory[lmk], sharedVars[1:],
                                    matchedValues + [lmk], results, switched)
        elif matchedValues:
            if leftMemory:
                for lm, ljust in keysToList(leftMemory):
                    #Both memories are nonempty
                    if rightMemory:
                        for rm, rjust in keysToList(rightMemory):
                            """if switched:
                                row = matchedValues + rm + lm
                            else:"""
                            row = matchedValues + lm + rm
                            results.append([row, []])
                            self.add(row, [])
                    else:
                        #Left memory is nonempty, right is empty
                        row = matchedValues + lm
                        results.append([row, []])
                        self.add(row, [])
            elif rightMemory:
                for rm, rjust in keysToList(rightMemory):
                    #Left memory is empty, right is nonempty
                    row = matchedValues + rm
                    results.append([row, []])
                    self.add(row, [])
        return results


    def getvar(self, var, fact):
        """I return values for var from fact"""
        values = list()
        if var in self.pattern:
            pos = self.pattern.index(var)
            return fact[pos]
        else:
            raise exception.UnboundRuleVariable(var, self.pattern)

    #####Make it a method of RuleNode
    def matchingFacts(self, rhs, facts):
        """I generate a set of facts and justifications, of the form:
        
        [[fact1, fact, fact2], [just1, just2, just2]]
        
        according to the given rhs from 'facts'"""
        results = list()
        for fact, just in facts:
            newFact = list()
            for p in rhs:
                if isinstance(p, Variable):
                    newFact.append(self.getvar(p, fact))
                else:
                    newFact.append(p)
#            print "The fact: ", Fact(*newFact), " is justified by: ", just
            results.append(Fact(*newFact))
        return results

    def __repr__(self):
        return """BetaNode(%s)(left: %s, right: %s)""" %(self.pattern, self.lnode, self.rnode)

class AlphaStore(object):
    def __init__(self):
        self.nodes = list()
        self.sharedIndex = dict() #records what variables each node contains
        self.py_lists = dict() # mapping from bnodes that are head of rdf lists to  python lists
        
    def addNode(self, node):
        from builtins import builtinp
        if node not in self.nodes:
            self.nodes.append(node)
            # Only index the non-builtins for sorting (since we want to make sure
            # builtins appear at the end
            if not builtinp(node) and not isListVarNode(node):                
                for v in node.vars:
                    if v not in self.sharedIndex:
                        self.sharedIndex[v] = [node]
                    else:
                        self.sharedIndex[v].append(node)

    def __getitem__(self, index):
        return self.nodes[index]


    def listToPython(self, id, alphaNodes, py_list):
        
        if not alphaNodes:
            return py_list
        
        node_first = [node for node in alphaNodes if convertBNodeToFact(node.pattern.s)==id and node.pattern.p==FIRST]

        #it's tricky here because the subj might be a variable
        if node_first:
            if isinstance(node_first[0].pattern.o, Exivar):
                return self.listToPython(convertBNodeToFact(node_first[0].pattern.o), alphaNodes,  py_list)
            elif isinstance(node_first[0].pattern.o, Variable):
                py_list.append(node_first[0].pattern.o) # append variable
            

        node_rest = [node for node in alphaNodes if convertBNodeToFact(node.pattern.s)==id and node.pattern.p==REST]
        if node_rest:
            if node_rest[0].pattern.o == NIL:      
                return py_list
            else:          
                return self.listToPython(convertBNodeToFact(node_rest[0].pattern.o), alphaNodes,  py_list)
        else:
            return py_list
        
    def generateDependencies(self):
        from builtins import funcBuiltinp
        for a in self.nodes:
            if funcBuiltinp(a):
                #means that object is a variable, look for dependencies on that var.
                for x in self.nodes:
                    if x != a and a.pattern.o in [x.pattern.s, x.pattern.o]:
                        a.dependents.append(x)

            #rdf:rest should always be before rdf:first                          
            elif a.pattern.p == REST:
                for x in self.nodes:
                    if  x.pattern.p == FIRST and x.pattern.s == a.pattern.s:
                        a.dependents.append(x)                                    
        
        for a in self.nodes:
            if funcBuiltinp(a):
                
                # (?x ?a) math:sum z. we want to have x and a  bound before evaluating math:sum.
                #so, look for the variables in the list, and see where they occur in the nodes set, then add dependencies                
                #first, generate the variables from the list
                vars = self.listToPython(convertBNodeToFact(a.pattern.s), self.nodes, [])

                #now add dependencies
                for x in self.nodes:
                    if x.pattern.p==FIRST and x.pattern.o in vars:
                        x.dependents.append(a)

        """                        
        for a in self.nodes:
            if a.dependents:

                pass                
                #print "a:", a
                #print "dependents:", a.dependents
            """
            
    """   1.  Select a vertex that has in-degree zero.
   2. Add the vertex to the sort.
   3. Delete the vertex and all the edges emanating from it from the graph. """
    def indegree(self, node, graph):
        #print "node", node
        for a in graph:
            if node in a.dependents:
                return 1
        return 0
            
    lastTopSort = -1000000000L
    def topSort(self, unsorted, sorted):
##        k = len(unsorted)
##        if k > self.lastTopSort:
##            import traceback
##            #traceback.print_stack()
##        self.__class__.lastTopSort = k
        from builtins import funcBuiltinp
##        if not unsorted:
##            #for i in sorted:
##            #   print "s", i            
##            return sorted
##        else:
##            first = [x for x in unsorted if self.indegree(x,unsorted)==0 and not funcBuiltinp(x)]            
##            if first:                
##                sorted.append(first[0])
##                unsorted.remove(first[0])
##                #print len(unsorted)
##                return self.topSort(unsorted, sorted)
##            else:               
##                first = [x for x in unsorted if self.indegree(x,unsorted)==0]                            
##                sorted.append(first[0])
##                unsorted.remove(first[0])
##                return self.topSort(unsorted, sorted)

        try: set
        except:
             from sets import Set as set
        unsorted = set(unsorted)
        inDegrees = {}
        for node in unsorted:
            inDegrees[node] = 0
        for node in unsorted:
            for parent in node.dependents:
                if parent in inDegrees:
                    inDegrees[parent] = inDegrees[parent] + 1
        zeros = set()
        simpleZeros = set()
        for node in inDegrees:
            if inDegrees[node] == 0:
                if funcBuiltinp(node):
                    zeros.add(node)
                else:
                    simpleZeros.add(node)
        while zeros or simpleZeros:
            if simpleZeros:
                top = simpleZeros.pop()
            else:
                top = zeros.pop()
            sorted.append(top)
            for node in top.dependents:
                if node in inDegrees:
                    inDegrees[node] = inDegrees[node] - 1
                    if inDegrees[node] == 0:
                        if funcBuiltinp(node):
                            zeros.add(node)
                        else:
                            simpleZeros.add(node)
        if inDegrees and max(inDegrees.values()) != 0:
            raise ValueError
        return sorted
            
           
    def sort(self):
        self.generateDependencies()
        
        from builtins import builtinp
        builtins = list()
        for node in self.nodes:
            if builtinp(node):
                builtins.append(node)
        builtins = removedups(builtins)

        # for nodes with pattern like (_:exivar FIRST ?x)  we want to make sure ?x's are bound when joining,
        #so add them at the end (like builtins)        
        listVarNodes = [node for node in self.nodes if node.pattern.p in [FIRST, REST, INCLUDES, TYPE, SEMANTICS]]        
        listVarNodes = removedups(listVarNodes)
                        
        nonBuiltins = [x for x in self.nodes if x not in builtins]
        nonBuiltins = [x for x in nonBuiltins if x not in listVarNodes]        
        
        # Sort the non-builtin alphas so that nodes that share variables (if any)
        # are adjacent
        sortedNonBuiltins = list()
        for nb in nonBuiltins:
            for v in nb.vars:
                nodesThatShare = self.sharedIndex.get(v)
                if nodesThatShare:
                    sortedNonBuiltins.extend(nodesThatShare)
                    sortedNonBuiltins.append(nb)
        # If there were any nodes that shared variables, use the sorted list of nodes
        if sortedNonBuiltins:
            nonBuiltins = removedups(sortedNonBuiltins)                

        #only sort builtins and list nodes (rdf:first, rest, etc.)
        #the other nodes should come first
        unsortedNodes = removedups(builtins + listVarNodes)

        
        
        # make full topological sort
        self.nodes = removedups(nonBuiltins + self.topSort(unsortedNodes, []))
        """
        print "nodes:"
        for i in self.nodes:
            print i
        print "\n"
        """
        
    def display(self):
        for n in self.nodes:
            print n.pattern

class BetaStore(object):
    def __init__(self):
        self.nodes = list()

    def addNode(self, node):
        if node not in self.nodes:
            self.nodes.append(node)

    def __getitem__(self, index):
        return self.nodes[index]
        
class RuleNode(object):
    """I am a node for storing rules."""
    def __init__(self, rule):
        #only store the right-hand side
        self.__rhs = rule.rhs
        self.__ts = rule.timestamp
        self.proofTrace = ProofTrace()

    def __repr__(self): return """RuleNode(%s)""" % (self.rhs)

    def getRhs(self): return self.__rhs
    def setRhs(self, val): self.__rhs = val
    def getTs(self): return self.__ts

    rhs = property(getRhs, setRhs, None, None)
    timestamp = property(getTs, None, None, None)

    def __eq__(self, other):
        return (self == other)

    def __hash__(self):
        return hash(tuple(self.rhs))

    def __neq__(self, other):
        return not self.__eq__(self, other)

class QueryRuleNode:
    """I provide an interface to all the rows in the BetaNode that I am connected to."""
    def __init__(self):
        self.lhs = []
        self.rhs = []
        self.betaNode = None
        self.rows = Set()
        
    def getRows(self):
        """I return the rows of the BetaNode, one row at a time."""
        rows = keysToList(self.betaNode.ind)
        for r in rows:
            yield r
    
# class NodeStorage(object):
#     def __init__(self):
#         """I store all the (alpha or beta) nodes of a particular kind"""
#         self.store = Set()
    
#     def add(self, node):
#         self.store.add(node)
#         return self.store

#     def __repr__(self):
#         return """%s""" %(self.store)

class AlphaIndex:
    """I index AlphaNodes by pattern.  The pattern is abstract, e.g. all
    patterns of the form (variable predicate variable) will be grouped together."""
    def __init__(self):
        self.ind = dict()

    def index(self, ind, node, varsk):
        
        if varsk: # Still work to be done
            cur = varsk.pop(0)  #pop(0) pops the first item off            
            if cur not in ind:
                # So we know the node doesn't exist
                if varsk:
                    if isinstance(cur, Variable):
                        #index variables as 'None'
                        cur = None
                    if cur in ind:
                        ind[cur] = ind[cur]
                    else:
                        ind[cur] = {}
                    return self.index(ind[cur], node, varsk)                        
                else: # At the bottom, and the node
                    if isinstance(cur, Variable):
                        cur = None
                    if cur not in ind:
                        ind[cur] = [node]
                    else:
                        if ind[cur]:
                            # if the very same alpha node is not already there
                            # then add it
                            if node not in ind[cur]:
                                ind[cur].append(node)
                            else:
                                return False
                        else:
                            ind[cur] = [node]
                    return True 
            else:
                if varsk: # Perhaps the node does exist
                    if isinstance(cur, Variable):
                        cur = None
                    return self.index(ind[cur], node, varsk)
                else:
                    # It definitely exists.
                    return False

    def match(self, pattern):
        """I return a list of matching alpha nodes for a given pattern"""
        nodesMatched = list()
        key = [pattern.p, pattern.o, pattern.s]        
        self.matchhelper(key, self.ind, nodesMatched)
        return nodesMatched

    def matchhelper(self, pattern, ind, nodesMatched):
        """I determine what alpha node(s) a certain pattern matches."""        
        if not pattern:            
            nodesMatched.extend(ind)
            return nodesMatched
        p = pattern[0]
        if None in ind:
            self.matchhelper(pattern[1:], ind[None], nodesMatched)
        if p in ind:            
            self.matchhelper(pattern[1:], ind[p], nodesMatched)
               
    def add(self, anode):
        #index by pattern, followed by object, followed by subj
        key = [anode.pattern.p, anode.pattern.o, anode.pattern.s]
        return self.index(self.ind, anode, key)                

class BetaIndex(list):
    pass
