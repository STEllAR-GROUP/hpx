#!/usr/bin/python
"""Please Just Plot.  Module for generating plots hassle-free.

This module provides operations to help plot data easily.  It is best when used
as a quick and dirty way to plot, as it becomes more tedious to use the more
particular one is about how a plot looks.  Focus is on the data and ease of 
use.

The only plotting program in the implementation so far is gnuplot.  Though
efforts to support many terminal types may be added, use of terminal postscript
is encouraged.  The module is currently in early development, so a lot has not
been implemented.  Look for the "IMP" tag within the documentation to note what
is on the backburner.  If you notice something is missing and would like to
notify me, e-mail me at cmichael@cct.lsu.edu with the name "pjplot" on the
subject line.

IMP: Document gnuplot parameters.

Some examples of usage:

 Simple:
  #Prints the given points to a file named "Test Plot.ps"
  d1 = Data2D("Test Data One", "X", "Y")
  d2 = Data2D("Test Data Two", "X", "Y")

  d1.point(0, 1)
  d1.point(1, 2)
  d1.point(2, 3)
  d2.point(0, 1)
  d2.point(1, 1)
  d2.point(2, 2)

  p = Gnuplot("Test Plot")
  p.data(d1, d2)
  p.plot()
  p.write()

  pWithLines = Gnuplot("Test Plot with Lines")
  pWithLines.dataStyle(d1, withs="lines")
  pWithLines.dataStyle(d2, withs="lines")
  pWithLines.plot()
"""

import re

from os import popen2

_GNUPLOT_TERMINAL_X11 = """\
set terminal x11 %(reset)s %(n)s %(enhanced)s %(font)s %(title)s \
%(persist)s %(raise_)s %(close)s\
"""
_GNUPLOT_TERMINAL_X11_DEFAULTS = {\
'reset'    : '',
'n'        : '',
'enhanced' : '',
'font'     : '',
'title'    : '',
'persist'  : '',
'raise_'   : '',
'close'    : ''}

_GNUPLOT_TERMINAL_POSTSCRIPT = """\
set terminal postscript %(mode)s %(enhanced)s %(color)s %(colortext)s \
%(solid)s %(dashlength)s %(linewidth)s %(duplexing)s %(rounded)s %(fontfile)s \
%(palfuncparam)s %(fontname)s %(fontsize)s\
"""
_GNUPLOT_TERMINAL_POSTSCRIPT_DEFAULTS = {\
'mode'     : 'landscape',
'enhanced' : '',
'color'    : 'color',
'colortext': '',
'solid'    : 'solid',
'dashlength': '',
'linewidth': '',
'duplexing': '',
'rounded'  : '',
'fontfile' : '',
'palfuncparam' : '',
'fontname' : '"Helvetica"',
'fontsize' : ''}

_GNUPLOT_BOXWIDTH = """\
set boxwidth %(boxwidth)s\
"""
_GNUPLOT_BOXWIDTH_DEFAULTS = {\
'boxwidth' : '0.25'}

_GNUPLOT_KEY = """\
set key %(on)s\
""" 
_GNUPLOT_KEY_DEFAULTS = {\
'on'       : 'on'}

_GNUPLOT_TITLE = """\
set title "%(titleText)s"\
"""
_GNUPLOT_TITLE_DEFAULTS = {\
'titleText' : ''}

_GNUPLOT_XLABEL = """\
set xlabel "%(label)s"\
"""
_GNUPLOT_XLABEL_DEFAULTS = {\
'label'     : ''}

_GNUPLOT_YLABEL = """\
set ylabel "%(label)s"\
"""
_GNUPLOT_YLABEL_DEFAULTS = {\
'label'     : ''}

_GNUPLOT_ZLABEL = """\
set zlabel "%(label)s"\
"""
_GNUPLOT_ZLABEL_DEFAULTS = {\
'label'     : ''}

_GNUPLOT_XRANGE = """\
set xrange [%(range)s]\
"""
_GNUPLOT_XRANGE_DEFAULTS = {\
'range'    : ''}

_GNUPLOT_YRANGE = """\
set yrange [%(range)s]\
"""
_GNUPLOT_YRANGE_DEFAULTS = {\
'range'    : ''}

_GNUPLOT_XTICS = """\
set xtics axis rotate by -30 (%(xticsList)s)\
"""
_GNUPLOT_XTICS_DEFAULTS = {\
'xticsList' : ''}

_GNUPLOT_XTICSLIST_INLINE = """\
"%(title)s" %(pos)s\
"""
_GNUPLOT_XTICSLIST_INLINE_DEFAULTS = {\
'title' : '',
'pos'   : ''}

_GNUPLOT_YTICS = """\
set ytics (%(yticsList)s)\
"""
_GNUPLOT_YTICS_DEFAULTS = {\
'yticsList' : ''}

_GNUPLOT_YTICSLIST_INLINE = """\
"%(title)s" %(pos)s\
"""
_GNUPLOT_YTICSLIST_INLINE_DEFAULTS = {\
'title' : '',
'pos'   : ''}



_GNUPLOT_PLOT = """\
plot %(plotList)s\
"""
_GNUPLOT_PLOT_DEFAULTS = {\
'plotList'  : ''}

_GNUPLOT_PLOTLIST_INLINE = """\
'-' title "%(title)s" with %(withs)s\
"""
_GNUPLOT_PLOTLIST_INLINE_DEFAULTS = {
'title'     : '',
'withs'      : 'points'}


class Data:
   """Information to be plotted."""

   def __init__(self, name, *labels):
      """The name must be provided and will be used to denote the data in the
         plot.  Optionally, the axis labels may be provided."""
      self.name = name
      self.points = []
      self.labels = labels

   def labelsFromTuple(self, labelTuple):
      """Set the axis names from a given tuple.  May be better in some cases as
         opposed to specifying the labels in the constructor."""
      self.labels = labelTuple

   def sort(self, axis):
      """Sorts points on an axis (of type int).  Useful if data are strings 
         rather than numerical."""
      self.points.sort(lambda p1,p2:cmp(p1[axis], p2[axis]))

   def pointsFromFile(self, fileName):
      pass

   def getMax(self, axis):
      max = None

      for point in self.points:
         if point[axis] > max:
            max = point[axis]

      return max

   def getType(self, axis):
      return type(self.points[0][axis])

   def normalize(self, axis, ceiling=1.0):
      """Normalizes the points on the given axis index with respect to the
         ceiling."""
      max = self.getMax(axis)

      for point in self.points:
         point[axis] = point[axis] * ceiling / max

   def __str__(self):
      """Prints the data.
         IMP: Strings must be printed in quotes."""
      outputString = ""

      for point in self.points:
         for component in point:
            outputString += component.__str__() + ' '
         outputString += '\n'

      return outputString[:-1]

   def strFromKey(self, keyDict):
      """Prints the data, using keyDict to replace the value of strings."""
      outputString = ""

      for point in self.points:
         for component in point:
            if type(component) == str:
               outputString += keyDict[component].__str__() + ' '
            else:
               outputString += component.__str__() + ' '
         outputString += '\n'

      return outputString[:-1]


class Data2D(Data):
   """2D data to be plotted."""

   def __init__(self, name, labelX=None, labelY=None):
      """The name must be provided and will be used to denote the data in plot.
         Optionally, the axis labels may be provided"""
      Data.__init__(self, name, labelX, labelY)

   def point(self, x, y):
      """Stores the given point."""
      self.points.append([x,y])

   def pointsFromFile(self, fileName, xType, yType):
      """Looks in a file and stores the points specified.  The data must be
         specified similarly to the gnuplot style, except strings may be
         specified by quotes.  Types must be specified for each column of 
         data.
         IMP: strings read from file."""
      dataStream = open(fileName, 'r').readlines()

      for line in dataStream:
         point = re.findall('\S+', line)
         self.point(xType(point[0]), yType(point[1]))


class Data3D(Data):
   """3D data to be plotted."""

   def __init__(self, name, labelX=None, labelY=None, labelZ=None):
      """The name must be provided and will be used to denote the data in plot.
         Optionally, the axis labels may be provided"""
      Data.__init__(self, name, labelX, labelY, labelZ)

   def point(self, x, y, z):
      """Stores the given point."""
      self.points.append([x,y,z])

   def pointsFromFile(self, fileName, xType, yType, zType):
      """Looks in a file and stores the points specified.  The data must be
         specified similarly to the gnuplot style, except strings may be
         specified by quotes.  Types must be specified for each column of 
         data.
         IMP: strings read from file."""
      dataStream = open(fileName, 'r').readlines()

      for line in dataStream:
         point = re.findall('\S+', line)
         self.point(xType(point[0]), yType(point[1]), zType(point[2]))


class DataND(Data):
   """"4-or-more-D data that may be sliced into 2D or 3D data for plotting."""

   def __init__(self, name):
      """The name must be provided and will be used to denote the data in 
         plot."""
      Data.__init__(self, name)

   def pointsFromFile(self, fileName, types):
      """Looks in a file and stores the points specified.  The data must be
         specified similarly to the gnuplot style, except strings may be
         specified by quotes.  Types must be specified for each column of 
         data.
         IMP: strings read from file."""
      dataStream = open(fileName, 'r').readlines()
      
      for line in dataStream:
         point = []
         typesIter = types.__iter__()
         componentList = re.findall('\S+', line)

         for component in componentList:
            type = typesIter.next()
            point.append(type(component))

         self.points.append(point)

   def slice2D(self, dim0, dim1):
      """Returns a Data2D slice specified by the two dimension indices."""
      slice = Data2D(self.name)

      for point in self.points:
         slice.point(point[dim0], point[dim1])

      return slice

   def slice3D(self, dim0, dim1, dim2):
      """Returns a Data3D slice specified by the three dimension indices."""
      slice = Data3D(self.name)

      for point in self.points:
         slice.point(point[dim0], point[dim1], point[dim2])

      return slice


class Plot:
   """A plot."""
   def __init__(self, name):
      """The name must be provided and will be used to name the plot."""
      self.name = name
      self.dataList = []
      self.dataItems = 0

   def data(self, *data):
      """Stores the given data objects for plotting."""
      self.dataList = list(data)
      self.dataItems = len(data)

   def dataFromList(self, dataList):
      """Stores the given list of data objects for plotting."""
      self.dataList = dataList
      self.dataItems = len(dataList)

   def dataSingle(self, data):
      """Adds the given data object to storage for plotting."""
      self.dataList.append(data)
      self.dataItems += 1


 
class Gnuplot(Plot):
   """Use gnuplot to generate a plot.
      IMP: 3D Plotting."""

   def __init__(self, name, gnuplotPath = "gnuplot"):
      """The name must be provided and will be used to name the plot.
         A path to gnuplot should be provided if not global."""
      Plot.__init__(self, name)
      self.gnuplotPath = gnuplotPath
      self.termPostscript()
      self.boxwidth()
      self.key()
      self.title(titleText = name)
      self.cXlabel = ""
      self.cYlabel = ""
      self.cZlabel = ""
      self.cXrange = ""
      self.cYrange = ""
      self.cPlot = ""
      self.cXtics = ""
      self.cYtics = ""

      self.xticLabels = {}
      self.yticLabels = {}
      self.styleList = []

   def _getString(template, defaults, params):
      paramDict = defaults.copy()
      paramDict.update(params)
      return template % paramDict
   _getString = staticmethod(_getString)

   def termX11(self, **params):
      """Used for testing purposes."""
      self.cTerm =  self._getString(_GNUPLOT_TERMINAL_X11,
                                    _GNUPLOT_TERMINAL_X11_DEFAULTS,
                                    params)

   def termPostscript(self, **params):
      self.ext = '.ps'
      self.cTerm = self._getString(_GNUPLOT_TERMINAL_POSTSCRIPT,
                                   _GNUPLOT_TERMINAL_POSTSCRIPT_DEFAULTS,
                                   params)
   def boxwidth(self, **params):
      self.cBoxwidth = self._getString(_GNUPLOT_BOXWIDTH,
                                       _GNUPLOT_BOXWIDTH_DEFAULTS, params)
   def key(self, **params):
      self.cKey = self._getString(_GNUPLOT_KEY, _GNUPLOT_KEY_DEFAULTS, params)

   def title(self, **params):
      self.cTitle = self._getString(_GNUPLOT_TITLE,
                                    _GNUPLOT_TITLE_DEFAULTS,params)

   def xlabel(self, **params):
      self.cXlabel = self._getString(_GNUPLOT_XLABEL,
                                     _GNUPLOT_XLABEL_DEFAULTS,
                                     params)

   def ylabel(self, **params):
      self.cYlabel = self._getString(_GNUPLOT_YLABEL,
                                     _GNUPLOT_YLABEL_DEFAULTS,
                                     params)

   def zlabel(self, **params):
      self.cZlabel = self._getString(_GNUPLOT_ZLABEL,
                                     _GNUPLOT_ZLABEL_DEFAULTS,
                                     params)

   def xrange(self, **params):
      self.cXrange = self._getString(_GNUPLOT_XRANGE,
                                     _GNUPLOT_XRANGE_DEFAULTS,
                                     params)

   def yrange(self, **params):
      self.cYrange = self._getString(_GNUPLOT_YRANGE,
                                     _GNUPLOT_YRANGE_DEFAULTS,
                                     params)

   def _tics(self, ticDict, index):
      ticIndex = 0

      for data in self.dataList:
         for point in data.points:
            if not ticDict.has_key(point[index]):
               ticDict[point[index]] = ticIndex
               ticIndex += 1


   def xtics(self, **params):
      """Readies the X axis for string data.
         IMPORTANT: Call this function if and only if you're using strings to
         specify X axis tics!"""
      if self.dataList[0].getType(0) == str:
         self._tics(self.xticLabels, 0)

      xticsList = []

      for xtic in self.xticLabels:
         xticsList.append(
               self._getString(_GNUPLOT_XTICSLIST_INLINE,
                               _GNUPLOT_XTICSLIST_INLINE_DEFAULTS,
                               {'title' : xtic, 'pos' : self.xticLabels[xtic]}))

      xticsListStr = ", ".join(xticsList)
      paramsWithXtics = {'xticsList' : xticsListStr}
      paramsWithXtics.update(params)

      self.cXtics = self._getString(_GNUPLOT_XTICS,
                                    _GNUPLOT_XTICS_DEFAULTS,
                                    paramsWithXtics)

   def ytics(self, **params):
      """Readies the Y axis for string data.
         IMPORTANT: Call this function if and only if you're using strings to
         specify Y axis tics!"""
      if self.dataList[0].getType(1) == str:
         self._tics(self.yticLabels, 1)

      yticsList = []

      for ytic in self.yticLabels:
         yticsList.append(
               self._getString(_GNUPLOT_YTICSLIST_INLINE,
                               _GNUPLOT_YTICSLIST_INLINE_DEFAULTS,
                               {'title' : ytic, 'pos' : self.yticLabels[ytic]}))

      yticsListStr = ", ".join(yticsList)
      paramsWithYtics = {'yticsList' : yticsListStr}
      paramsWithYtics.update(params)
      
      self.cYtics = self._getString(_GNUPLOT_YTICS,
                                    _GNUPLOT_YTICS_DEFAULTS,
                                    paramsWithYtics)


   def plot(self):
      """Plots the data.  The plot is stored internally to the object and can be
         printed to a file using the write() method."""
      plotList = []
      append = []
      styleListIter = self.styleList.__iter__()

      for data in self.dataList:
         if not self.styleList == []:
            currentStyle = styleListIter.next()
         else:
            currentStyle = {}

         params = {'title' : data.name}
         params.update(currentStyle)
         keyDict = self.xticLabels.copy()
         keyDict.update(self.yticLabels)
         append.append(data.strFromKey(keyDict))
         plotList.append(self._getString(_GNUPLOT_PLOTLIST_INLINE,
                                         _GNUPLOT_PLOTLIST_INLINE_DEFAULTS,
                                         params))

      plotListStr = ", ".join(plotList)
      appendStr = "\ne\n".join(append)

      self.cPlot = self._getString(_GNUPLOT_PLOT,
                                   _GNUPLOT_PLOT_DEFAULTS,
                                   {'plotList':plotListStr})
      self.cPlot += "\n" + appendStr

   def _setLabels(self, labels):
      numLabels = len(labels)

      if numLabels <= 2:
         self.xlabel(label = labels[0])
         self.ylabel(label = labels[1])
      
      if numLabels == 3:
         self.zlabel(label = labels[2])

   def data(self, *data):
      """Stores the given data objects for plotting."""
      Plot.dataFromList(self, list(data))
      labels = data[0].labels
      self._setLabels(labels)

   def dataStyle(self, data, **params):
      """Adds the given data object to storage for plotting while setting its
         plot style."""
      Plot.dataSingle(self, data)
      labels = data.labels
      self._setLabels(labels)
      self.styleList.append(params)

   def __str__(self):
      return self.cTerm + "\n" + self.cBoxwidth + "\n" + self.cKey + "\n" + \
             self.cTitle + "\n" + \
             self.cXtics + "\n" + self.cYtics + "\n" + \
             self.cXlabel + "\n" + self.cYlabel + "\n" + self.cZlabel + "\n"+ \
             self.cXrange + "\n" + self.cYrange + "\n" + \
             self.cPlot

   def write(self, fileName = None):
      """Writes the generated plot to a file.  If no filename is specified, one
         will be created similar to the plot's name.  Note that the plot()
         method must be called before doing this."""
      if fileName == None:
         fileName = self.name.replace('/', '(over)')

      (stdin, stdout) = popen2(self.gnuplotPath)

      stdin.write(self.__str__())
      stdin.close()

      outputFileName = fileName + self.ext

      outputFile = open(outputFileName, 'w')
      outputFile.write(stdout.read())
      outputFile.close()
      stdout.close()
      #print outputFileName

if __name__ == '__main__':
   d1 = Data2D("Test Data", "Bits", "Frequency")
   d1.point(0, 3)
   d1.point(1, 10)
   d1.point(2, 400)

   d2Labels = ("Bits", "Frequency")
   d2 = Data2D("More Test Data")
   d2.labelsFromTuple(d2Labels)
   d2.point(0, 8)
   d2.point(1, 19)
   d2.point(2, 400)
   d2.point(3, 450)

   p = Gnuplot("Test Plot")
#   p.data(d1, d2)
   p.termPostscript()
   p.dataStyle(d1, withs='lines')
   p.dataStyle(d2, withs='points')
#   p.xtics()
#   p.ytics()
   p.plot()
   p.write()
