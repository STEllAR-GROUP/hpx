#
#
#
# TODO:
# - extraction os fields should extend date time if necessary
#
# See http://www.python.org/doc/current/lib/module-time.html
#

"""
The time built-ins concern dates and times expressed in a specific
version of ISO date-time format.  These functions allow the various
parts of the date and time to be compared, and converted
into interger second GMT era format for arithmetic.

Be aware that ISo times carry timezone offset information:  they cannot be
converted to integer second times without a valid timezone offset, such as "Z".
"""

import re
import notation3    # N3 parsers and generators, and RDF generator
import isodate      # Local, by mnot. implements <http://www.w3.org/TR/NOTE-datetime>


from diag import progress, verbosity
from term import LightBuiltIn, Function, ReverseFunction
import time, calendar # Python standard distribution


TIMES_NS_URI = "http://www.w3.org/2000/10/swap/times#"


__version__ = "0.3"

DAY = 24 * 60 * 60

class BI_inSeconds(LightBuiltIn, Function, ReverseFunction):
    """For a time string, the number of seconds from the era start as an integer-representing string.
    """
    def evaluateObject(self, subj_py):
        try:
            return str(isodate.parse(subj_py))
        except:
            return None

    def evaluateSubject(self, obj_py):
        return isodate.fullString(int(obj_py))

class BI_equalTo(LightBuiltIn):
    def evaluate(self, subj_py, obj_py):
        try:
            return isodate.parse(subj_py) == isodate.parse(obj_py)
        except:
            return None

class BI_year(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        try:
            return subj_py[:4]
        except:
            return None

class BI_month(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        try:
            return subj_py[5:7]
        except:
            return None

class BI_day(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        try:
            return subj_py[8:10]
        except:
            return None

class BI_date(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        try:
            return subj_py[:10]
        except:
            return None

class BI_hour(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        try:
            return subj_py[11:13]
        except:
            return None

class BI_minute(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        try:
            return subj_py[14:16]
        except:
            return None

class BI_second(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        try:
            return subj_py[17:19]
        except:
            return None

tzone = re.compile(r'.*([-+]\d{1,2}:\d{2,2})')
class BI_timeZone(LightBuiltIn, Function):
    def evaluateObject(self,  subj_py):
        m = tzone.match(subj_py)
        if m == None: return None
        return m.group(1)

class BI_dayOfWeek(LightBuiltIn, Function):
    def evaluateObject(self,  subj_py):
        weekdayZero = time.gmtime(0)[6]
        return str((weekdayZero + int(isodate.parse(subj_py)/DAY)) % 7 )


#
class BI_format(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        """params are ISO time string, format string. Returns reformatted. Ignores TZ@@"""
        if verbosity() > 80: progress("strTime:format input:"+`subj_py`)
        str, format = subj_py
        try:
            return  time.strftime(format, time.gmtime(isodate.parse(str)))
        except:
            return None

#
class BI_gmTime(LightBuiltIn, Function):
    def evaluateObject(self,  subj_py):
        """Subject is  (empty string for standard formatting or) format string.
        Returns formatted."""
        if verbosity() > 80: progress("time:gmTime input:"+`subj_py`)
        format = subj_py
        if format =="" : format="%Y-%m-%dT%H:%M:%SZ"
        try:
            return time.strftime(format, time.gmtime(time.time()))
        except:
            return isodate.asString(time())

class BI_localTime(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        """Subject is format string or empty string for std formatting.
        Returns reformatted. @@@@ Ignores TZ"""
        if verbosity() > 80: progress("time:localTime input:"+`subj_py`)
        format = subj_py
        if format =="" : return   isodate.asString(time.time())
        return   time.strftime(format, time.localtime(time.time()))



#  original things from mNot's cwm_time.py:
#
#  these ise Integer time in seconds from epoch.
#
class BI_formatSeconds(LightBuiltIn, Function):
    def evaluateObject(self,  subj_py):
        """params are epoch-seconds time string, format string. Returns reformatted"""
        if verbosity() > 80: progress("strTime:format input:"+`subj_py`)
        str, format = subj_py
        try:
            return  time.strftime(format, time.gmtime(int(str)))
        except:
            return None

class BI_parseToSeconds(LightBuiltIn, Function):
    def evaluateObject(self,   subj_py):
        if verbosity() > 80: progress("strTime:parse input:"+`subj_py`)
        str, format = subj_py
        try:
            return  str(calendar.timegm(time.strptime(str, format)))
        except:
            return None




#  Register the string built-ins with the store
def register(store):
    str = store.symbol(TIMES_NS_URI[:-1])
    str.internFrag("inSeconds", BI_inSeconds)
    str.internFrag("year", BI_year)
    str.internFrag("month", BI_month)
    str.internFrag("day", BI_day)
    str.internFrag("date", BI_date)
    str.internFrag("equalTo", BI_equalTo)
    str.internFrag("hour", BI_hour)
    str.internFrag("minute", BI_minute)
    str.internFrag("second", BI_second)
    str.internFrag("dayOfWeek", BI_dayOfWeek)
    str.internFrag("timeZone", BI_timeZone)
    str.internFrag("gmTime", BI_gmTime)
    str.internFrag("localTime", BI_localTime)
    str.internFrag("format", BI_format)
#    str.internFrag("parse", BI_parse)
    str.internFrag("formatSeconds", BI_formatSeconds)  # Deprocate?
    str.internFrag("parseToSeconds", BI_parseToSeconds)  # Deprocate?


# ends
