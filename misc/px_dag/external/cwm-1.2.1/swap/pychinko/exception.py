class UnboundRuleVariable(Exception):
    def __init__(self, var, pattern):
        Exception.__init__(self, """%s is not bound in %s""" % (var, pattern))

class UnknownN3Type(Exception):
    def __init__(self, type, value):
        Exception.__init__(self, """%s is not a known N3 type (value was %s)"""
                           % (type, value))

class AssertError(Exception):
    """I am raised when the validation predicate 'assert' fails."""
    def __init__(self, subj, pred, obj):
        Exception.__init__(self, """Fact(%s, %s, %s) is not in the fact base."""
                           % (subj, pred, obj))

class ReportError(Exception):
    """I am raised when the validation predicate 'report' fails."""
    def __init__(self, subj, pred, obj):
        Exception.__init__(self, """Fact(%s, %s, %s) is in the fact base."""
                           % (subj, pred, obj))
        
class UnknownBuiltin(Exception):
    """I am raised when an unknown builtin is referenced."""
    def __init__(self, builtin):
        Exception.__init__(self, "Unknown builtin '%s' referenced."
                           % (builtin))

class UnknownFactsFile(Exception):
    """I am raised when an unknown facts file format is passed in."""
    def __init__(self, filename):
        Exception.__init__(self, "Unknown facts file '%s'"
                           % (filename))
