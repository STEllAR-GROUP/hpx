"""Utility functions for ordered seqeunces

When you are dealing with sequences which you know are ordered,
then set operations are linear instead of square in the length of the
sequences. This makes an ordered sequence a practical representation of a set.

@@ Now python has sets, these should be used where the ordering is not otherwise
necessary.  @@ check 
$Id: OrderedSequence.py,v 1.2 2007/06/26 02:36:15 syosi Exp $
"""

def merge(a,b):
    """Merge sorted sequences

    The fact that the sequences are sorted makes this faster"""
    i = 0
    j = 0
    m = len(a)
    n = len(b)
    result = []
    while 1:
        if i==m:   # No more of a, return rest of b
            return result + b[j:]
        if j==n:
            return result + a[i:]
        if a[i] < b[j]:
            result.append(a[i])
            i = i + 1
        elif a[i] > b[j]:
            result.append(b[j])
            j = j + 1
        else:  # a[i]=b[j]
            result.append(a[i])
            i = i + 1
            j = j + 1
        
def intersection(a,b):
    """Find intersection of sorted sequences

    The fact that the sequences are sorted makes this faster"""
    i = 0
    j = 0
    m = len(a)
    n = len(b)
#    progress(" &&& Intersection of %s and %s" %(a,b))
    result = []
    while 1:
        if i==m or j==n:   # No more of one, return what we have
            return result
        if a[i] < b[j]:
            i = i + 1
        elif a[i] > b[j]:
            j = j + 1
        else:  # a[i]=b[j]
            result.append(a[i])
            i = i + 1
            j = j + 1
    
def minus(a,b):
    """Those in a but not in b for sorted sequences

    The fact that the sequences are sorted makes this faster"""
    i = 0
    j = 0
    m = len(a)
    n = len(b)
    result = []
#    progress(" &&& minus of %s and %s" %(a,b))
    while 1:
        if j==n:   # No more of b, return rest of a
            return result + a[i:]
        if i==m:   # No more of a, some of b - error
            raise ValueError("Cannot remove items" + `b[j:]`)
            return result + b[j:]
        if a[i] < b[j]:
            result.append(a[i])
            i = i + 1
        elif a[i] > b[j]:
            raise ValueError("Cannot remove item" + `b[j]`)
            j = j + 1
        else:  # a[i]=b[j]
            i = i + 1
            j = j + 1
        
#______________________________________________ More utilities

def indentString(str):
    """ Return a string indented by 4 spaces"""
    s = "    "
    for ch in str:
        s = s + ch
        if ch == "\n": s = s + "    "
    if s.endswith("    "):
        s = s[:-4]
    return s


