#!/usr/bin/env python

"""
isodate.py

Copyright 2002 Mark Nottingham, <mailto:mnot@pobox.com>

    THIS SOFTWARE IS SUPPLIED WITHOUT WARRANTY OF ANY KIND, AND MAY BE
    COPIED, MODIFIED OR DISTRIBUTED IN ANY WAY, AS LONG AS THIS NOTICE
    AND ACKNOWLEDGEMENT OF AUTHORSHIP REMAIN.

Functions for manipulating a subset of ISO8601 date, as specified by
  <http://www.w3.org/TR/NOTE-datetime>
  
Exposes:
  - parse(s)
    s being a conforming (regular or unicode) string. Raises ValueError for
    invalid strings. Returns a float (representing seconds from the epoch; 
    see the time module).
    
  - asString(i)
    i being an integer or float. Returns a conforming string.
  
TODO:
  - Precision? it would be nice to have an interface that tells us how
    precise a datestring is, so that we don't make assumptions about it; 
    e.g., 2001 != 2001-01-01T00:00:00Z.
    
    2002-06-22 added bad string to error message -- timbl@w3.org
"""

import sys, time, re, operator
import calendar # timegm - from python 

from types import StringType, UnicodeType, IntType, LongType, FloatType

__version__ = "0.6"
date_parser = re.compile(r"""^
    (?P<year>\d{4,4})
    (?:
        -
        (?P<month>\d{1,2})
        (?:
            -
            (?P<day>\d{1,2})
            (?:
                T
                (?P<hour>\d{1,2})
                :
                (?P<minute>\d{1,2})
                (?:
                    :
                    (?P<second>\d{1,2})
                    (?:
                        \.
                        (?P<dec_second>\d+)?
                    )?
                )?                    
                (?:
                    Z
                    |
                    (?:
                        (?P<tz_sign>[+-])
                        (?P<tz_hour>\d{1,2})
                        :
                        (?P<tz_min>\d{2,2})
                    )
                )
            )?
        )?
    )?
$""", re.VERBOSE)


def parse(s):
    """ parse a string and return seconds since the epoch. """
    assert type(s) in [StringType, UnicodeType]
    r = date_parser.search(s)
    try:
        a = r.groupdict('0')
    except:
        raise ValueError, 'invalid date string format:'+s
    y = int(a['year'])
    if y < 1970:
        raise ValueError, 'Sorry, date must be in Unix era (1970 or after):'+s
    d = calendar.timegm((   int(a['year']), 
                        int(a['month']) or 1, 
                        int(a['day']) or 1, 
                        int(a['hour']), 
                        int(a['minute']),
                        int(a['second']),
                        0,
                        0,
                        0
                    ))
    return d - int("%s%s" % (
            a.get('tz_sign', '+'), 
            ( int(a.get('tz_hour', 0)) * 60 * 60 ) - \
            ( int(a.get('tz_min', 0)) * 60 ))
    )
    
def fullString(i):
    """ given seconds since the epoch, return a full dateTime string in Z timezone. """
    assert type(i) in [IntType, FloatType, LongType], "Wrong type: "+ `type(i)` +`i`
    year, month, day, hour, minute, second, wday, jday, dst = time.gmtime(i)
    return str(year) + '-%2.2d-%2.2dT%2.2d:%2.2d:%2.2dZ' % (month, day, hour, minute, second)


def asString(i):
    """ given seconds since the epoch, return a dateTime string. """
    assert type(i) in [IntType, FloatType]
    year, month, day, hour, minute, second, wday, jday, dst = time.gmtime(i)
    o = str(year)
    if (month, day, hour, minute, second) == (1, 1, 0, 0, 0): return o
    o = o + '-%2.2d' % month
    if (day, hour, minute, second) == (1, 0, 0, 0): return o
    o = o + '-%2.2d' % day
    if (hour, minute, second) == (0, 0, 0): return o
    o = o + 'T%2.2d:%2.2d' % (hour, minute)
    if second != 0:
        o = o + ':%2.2d' % second
    o = o + 'Z'
    return o
