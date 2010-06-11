#!/usr/bin/python
"""
$Id: TableRenderer.py,v 1.3 2007/06/26 02:36:16 syosi Exp $

TableRenderer - spit out data in a space-extended ascii table

SYNOPSIS

 import TableRenderer
 a = [[1, 'a'], [2, 'b']]
 renderer = TableRenderer()
 renderer.addHeaders("int", "char")
 renderer.addData(a)
 print renderer.toString()+"\n"

DESCRIPTION

Display tabular data in a space-extended ascii table.

METHODS

    TableRenderer() - Create a new TableRenderer. Takes arguments in hash or array form.

        -dataFilter - Specify code to call to manipute the data. This code should alter C<$_> to appear as it should in the serialized ascii table.
        -headerFilter - Specify code to call to manipute the header labels. This code should alter C<$_> to appear as it should in the serialized ascii table.

    addHeader - Add headers to the current TableRenderer data set. This is customarily done at the beginning, however, headers may be added anywhere in the data. Headers added after data will, in fact, appear after that data.

        The headers may be passed in as a list, an array ref, or an array of arrays. The following are all equivilent:

        renderer.addHeaders('letters', 'numbers')
        renderer.addHeaders(['letters', 'numbers'])
        renderer.addHeaders([['letters', 'numbers']])
        renderer.addData('letters', 'numbers'), self.underline()

See FORMATTING for affects of linefeeds in the data.

    addData - Add data to the current TableRenderer data set. The data may be passed in as a list, an array ref, or an array of arrays (see addHeader). See FORMATTING for affects of linefeeds in the data.

    underline ([index]) - Underline (highlight) the last row of data. If a numeric C<$index> is provided, underline only that column.

    toString - Serialize the added headers and data in an ascii table.

FORMATTING

Cell entries may have line feeds C<"\n"> embedded in them. In this case, the data is rendered on multiple lines, but the tabular format is preserved.

Row data may have more columns than previous headers and data. This will have no affect accept that subsequent calls to L<underline> will underline the extra column.

TESTING

The test included in __main__ should produce this output:

    +------+------+------+
    |    my|  your|      |
    |letter|number|      |
    |------|------|      |
    |     a|     1|      |
    |------|------|      |
    |     b|     2| extra|
    |      |      |column|
    |------|------|------|
    |     c|     3|      |
    |     d|     4|      |
    |      |------|      |
    +------+------+------+
"""

# Classes of data cells
DATUM = 1       # normal
HEADER = 2      # underlined

import string
from diag import progress # @@ for debugging

def Assure(list, index, value):
    for i in range(len(list), index+1):
        list.append(value)

def _x(char, width):
    ret = ''
    for i in range(width):
        ret = ret+char
    return ret

class TableRenderer:

    def __init__(self, firstTitle=None, headerFilter=None, dataFilter=None):
        self.headerFilter = headerFilter
        self.dataFilter = dataFilter
        self.data = []          # [rows x columns] matrix of added data
        self.types = []         # type (DATUM or HEADER) of each row
        self.widths = []        # widths of each row
        self.heights = []       # heights of each row
        if (firstTitle):
            self.addHeaders(firstTitle)

    def addHeaders(self, data):
        self._addData(data, HEADER, self.headerFilter)

    def addData(self, data):
        self._addData(data, DATUM, self.dataFilter)

    def underline(self, index=None):
        if (index):
            self.data[-1][index][1] = HEADER
        else:
            for column in self.data[-1]:
                column[1] = HEADER
    def toString(self):
        ret = []
        ret.append(self._hr())
        for rowNo in range(len(self.data)):
            row = self.data[rowNo]
            ret.append(string.join(self._renderRow(self.data[rowNo], self.heights[rowNo]), "\n"))
        ret.append(self._hr())
        return string.join(ret, "\n")

    def _addData(self, data, theType, filter):
        progress("@@ _addData data=%s theType=%s filter=%s " %(data, theType, filter))
        progress("@@@ type ", type(data))
        if type(data[0]) is not type([]): data = [data]   # resolve ambiguous calling convention
        try:
            data[0].isdigit() # Lord, there's got to be a better way. @@@   (@@ to do what? --tim)
            data = [data]
        except AttributeError, e:
            data = data  #  @@? what does this do - tim   mean "pass"?
        for row in data:
            rowEntry = []
            self.data.append(rowEntry)
            self.types.append(DATUM)
            self.heights.append(0)
            for columnNo in range(len(row)):
                if (columnNo == len(self.widths)):
                    self.widths.append(0)
                datum = str(row[columnNo])
                if (filter):
                    datum = filter(datum)
                self._checkWidth(datum, len(self.data) - 1, columnNo)
                rowEntry.append([datum, theType])

    def _hr(self):
        ret = []
        for width in self.widths:
            ret.append(_x('-', width))
        ret[:0] = ['']
        ret.append('')
        return string.join(ret, '+')

    def _checkWidth(self, datum, rowNo, column):
        lines = string.split(str(datum), "\n")
        for line in lines:
            if (len(line) > self.widths[column]):
                self.widths[column] = len(line)
        if (len(lines) > self.heights[rowNo]):
            self.heights[rowNo] = len(lines)

    def _renderRow(self, row, height):
        footer = []
        rows = []
        needsFooter = 0
        for column in range(len(self.widths)):
            if (column == len(row)):
                row.append(['', DATUM])
            width = self.widths[column]
            datum = row[column]
            lines = string.split(datum[0], "\n")
            lineNo = 0
            while (lineNo < len(lines)):
                line = lines[lineNo]
                Assure(rows, lineNo, [])
                Assure(rows[lineNo], column, '')
                rows[lineNo][column] = _x(' ', (width - len(line))) + line
                lineNo = lineNo+1
            while (lineNo < height):
                Assure(rows, lineNo, [])
                Assure(rows[lineNo], column, ' ')
                rows[lineNo][column] = _x(' ', width)
                lineNo = lineNo+1
            if (datum[1] == HEADER):
                footer.append(_x('-', width))
                needsFooter = needsFooter+1
            else:
                footer.append(_x(' ', width)) # might need one on another column
        if (needsFooter):
            rows.append(footer)
        ret = []
        for row in rows:
            row[:0] = ['']
            row.append('')
            ret.append(string.join(row, '|'))
        return ret

if __name__ == '__main__':

    import re
    def df(datum):
        return re.sub("trimMe", "", datum)

    # Any data starting with "trimMe" will have that part removed.
    # df = None # lamda a: re.sub("trimMe", "", a)
    renderer = TableRenderer(dataFilter = df)

    # Add two two-line headers
    renderer.addHeaders(["my\nletter", "your\nnumber"])

    # Add a row of data that's just crying out to be filtered.
    renderer.addData(['trimMea', 'trimMe1'])

    # Underline that last row.
    renderer.underline()

    # Add another row, this time with an unexpected extra column ...
    renderer.addData(['trimMeb', 'trimMe2', "extra\ncolumn"])
    # ... and underline it.
    renderer.underline()

    # Add a two dimensional matrix of data.
    renderer.addData([['c', 3], ['d', 4]])

    # Underline just column 1 of the previous row.
    renderer.underline(index=1)

    # Let's see what hath we wrought.
    print renderer.toString()+"\n"
