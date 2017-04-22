import optparse
import math
import numpy
import itertools
from numpy import array
import matplotlib

#----------------------------------------------------------------------------
# Arguments and help
parser = optparse.OptionParser()
parser.add_option("--fig-size", type = "string", default = None)
parser.add_option("--show", action = "store_true", dest = "show_graph", default = False)
parser.add_option("--verbose", action = "store_true", default = False)
parser.add_option("--quiet", action = "store_true", default = False)
parser.add_option("--title", action = "store", dest = "title", default = False)

options, args = parser.parse_args();

#----------------------------------------------------------------------------
# convenience definitions to loop over all marker/colour styles
# if we have a lot of lines on the same graph
colours = ('r','g','b','c','y','m','k')
markers = ('+', '.', 'o', '*', '^', 's', 'v', ',', '<', '>', '8', 's', 'p', 'h', 'H', 'D', 'd')
filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')
#----------------------------------------------------------------------------

if (not options.show_graph) :
    matplotlib.use('SVG')
# this import must come after the use() call above
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
# setup default size of plot if user didn't pass fig_size command line option
try :
    size = list(map(float, options.fig_size.split(',')))
    if len(size) == 2 :
        print("found size ", size)
        options.fig_size = (size, [0.1, 0.1, 0.85, 0.85])
    elif len(size) == 6 :
        options.fig_size = (size[0:2], size[2:6])
    else :
        raise ValueError("--fig-size must be a string of 2 or 6 numbers")
except :
    options.fig_size = ([12, 9], [0.08, 0.14, 0.91, 0.83])
#    options.fig_size = ([6, 8], [0.16, 0.22, 0.79, 0.77])

#----------------------------------------------------------------------------
def maximum(iterable, default):
  '''Like max(), but returns a default value if xs is empty.'''
  try:
      return max(iterable)
  except ValueError:
       return default

#----------------------------------------------------------------------------
def minimum(iterable, default):
  '''Like min(), but returns a default value if xs is empty.'''
  try:
      return min(iterable)
  except ValueError:
       return default

#----------------------------------------------------------------------------
def sizeof_bytes(num):
   '''Output a number as human readable bytes.'''
   for x in ['bytes','KB','MB','GB','TB']:
        if num < 1024.0:
            return "%.0f %s" % (num, x)
        num /= 1024.0

#----------------------------------------------------------------------------
# plot N series of data onto a single graph
# each series is a array, there are N arrays in the supplied map
# graph_map, a map of arrays of {x,y,other} data
# labelstrings, {xaxis, yaxis, series_variable}
def plot_one_collection(graph_map, labelstrings, axes, axisfunction, minmax) :
    print("Plotting %i series of '%s'" % (len(graph_map), labelstrings[2]))

    # for convenience/brevity get the base, min, max for each exis
    xb = minmax[0][0]
    x1 = minmax[0][1]
    x2 = minmax[0][2]
    xm = minmax[0][3]
    yb = minmax[1][0]
    y1 = minmax[1][1]
    y2 = minmax[1][2]
    ym = minmax[1][3]

    # restart markers and colours from beginning of list for each new graph
    localmarkers = itertools.cycle(markers)
    localcolours = itertools.cycle(colours)
    series_keys = sorted(graph_map.keys())
    num_series = len(series_keys)
    for index in range(len(series_keys)):
        key = series_keys[index]
        series = sorted(graph_map[key])
        #print "The series is ", series
        # we can just plot the series directly, but just in case we add support
        # for error bars etc and use {x,y,stddev,etc...} in future, we will pull out
        # the values for plotting manually.
        values = [[v[0],v[1]] for v in series]
        #print "the values are ", values
        if (xb==0) and (yb==0):
            axes.plot(*list(zip(*values)), markersize=8, marker=next(localmarkers), color=next(localcolours))
        elif (xb!=0) and (yb==0):
            axes.semilogx(*list(zip(*values)), basex=xb, markersize=8, marker=next(localmarkers), color=next(localcolours))
        elif (xb==0) and (yb!=0):
            axes.semilogy(*list(zip(*values)), basey=yb, markersize=8, marker=next(localmarkers), color=next(localcolours))
        elif (xb!=0) and (yb!=0):
            axes.loglog(*list(zip(*values)), basex=xb, basey=yb, markersize=8, marker=next(localmarkers), color=next(localcolours))
        else:
          print("Error, unsupported log/lin options")

    # generate labels for each power of N on the axes
    if (xb!=0):
      # generate a list of numbers for the grid marks
      xlabels = tuple(i for i in (xb**x for x in range(x1,x2+1)) )
      # setup the xaxis parameters
      axes.set_xlim(minimum(xlabels,1)*(1.0-xm), maximum(xlabels,3)*(1.0+xm))
      axes.set_xticklabels(xlabels)
      axes.set_xscale('log', basex=xb)
      axes.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(axisfunction))
      axes.set_xlabel(labelstrings[0])
      axes.tick_params(axis='x', which='major', labelsize=9)
      axes.tick_params(axis='x', which='minor', labelsize=8)
    else:
      axes.set_xlim(x1, x2)

    if (yb!=0):
      # generate a list of numbers for the grid marks
      ylabels = tuple(i for i in (yb**y for y in range(y1,y2+1)) )
      # setup the yaxis parameters
      axes.set_ylim(minimum(ylabels,1)*(1.0-ym), maximum(ylabels,3)*(1.0+ym))
      axes.set_yticklabels(ylabels)
      axes.set_yscale('log', basey=yb)
      axes.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str('%.2f' % x)))
      axes.set_ylabel(labelstrings[1])
      axes.tick_params(axis='y', which='major', labelsize=9)
      axes.tick_params(axis='y', which='minor', labelsize=8)
    else:
      axes.set_ylim(y1, y2)
    axes.xaxis.grid(True)
    axes.yaxis.grid(True)


    #
    # define some custom minor tick locations on y axis:
    #
    #axes.yaxis.set_minor_formatter(plt.FormatStrFormatter('%8.4d'))
    #axes.yaxis.set_minor_locator(plt.FixedLocator([0.375,0.75,1.5,3,6,12]))
    #axes.tick_params(axis='y', which='minor', labelsize=8)
    #axes.grid(b=True, which='major', color='b', linestyle='-')
    #axes.grid(b=True, which='minor', color='g', linestyle='--')
    # coordinates are window coordinates from 0 to 1
    axes.set_title(labelstrings[2], fontsize=10)

#----------------------------------------------------------------------------
def plot_configuration(graph_map, mapnames, axesnames, titlefunction, legendfunction, legendtitlefunction, axisfunction, minmax, bbox) :

    fig = plt.figure(figsize = options.fig_size[0])
    axes = []

    # the supplied graphs come as a 2D array of params
    num_param1 = len(list(graph_map.keys()))
    num_param2 = len(list(graph_map[list(graph_map.keys())[0]].keys()))

    # All the graphs of param2 will be the same type,
    # but we need one legend per param1 regardless
    # so num_param2legend is used in places to add space for the extra legend plot
    num_param2legend = num_param2+1
    doLegend         = True
    numrows          = num_param1
    numcols          = num_param2legend

    # if the arrays is 1xN or Nx1, rearrange the num rows/cols
    # to fit the page a little better instead of having one long row/column of plots
    rearranged = False
    if (num_param1==1) or (num_param2==1):
      total = num_param1*num_param2legend
      print("total is ", total)
      better = int(math.sqrt(total))
      numrows = better
      numcols = int(math.ceil(total/float(better)))
      rearranged = True
      print("Rearranged graphs from %i x %i using layout %i x %i" % (num_param1, num_param2, numrows, numcols))

    # create an array of graphs for our parameter space
    # grid cells are defined by {row, col} from top left and down
    print("Creating array of graphs rows %i, cols %i" % (numrows, numcols))
    plot_index = 0
    row = 0
    col = 0
    graph_keys = sorted(graph_map.keys())
    for param1_i in range(num_param1):
      param1_key     = graph_keys[param1_i]
      param1_results = graph_map[param1_key]
      param1_keys    = sorted(param1_results.keys())
      print("param1_ type ", param1_key)

      # The legend must cover all graphs, not just the final one plotted
      legend_entries = []

      for param2_i in range(num_param2):
        newplot = plt.subplot2grid((numrows, numcols), (row, col), colspan=1)
        axes.append( newplot )
        try:
          print("num params %i and keys" % num_param2, param1_keys)
          param2_key     = param1_keys[param2_i]
          param2_results = param1_results[param2_key]
          param2_keys    = sorted(param2_results.keys())
          print("param2_ type ", param2_key)
          print("generating plot at {%i,%i}" % (row, col))
          plot_one_collection(param2_results,
            [axesnames[0], axesnames[1], mapnames[1] + " " + titlefunction(param2_key)],
            newplot,axisfunction, minmax)

          # merge lists for the legend
          legend_entries = list(set(legend_entries) | set(param2_keys))

        except:
          print("Failed to plot {%i,%i}" % (row, col))
        col += 1
        if ((col % numcols)==0):
          col = 0
          row += 1

      legend_entries = sorted(legend_entries)
      # at the end of each param2 group, there should be a legend
      leg = plt.subplot2grid((numrows, numcols), (row, col), colspan=1)
      leg.axis('off')
      leg.set_title(legendtitlefunction(param1_key))
      print("Legend title removed ")
      #leg.set_title(graph_keys[param1_i], fontsize=11)
      axes.append( leg )
      # restart markers and colours from beginning of list for each new graph
      localmarkers = itertools.cycle(markers)
      localcolours = itertools.cycle(colours)
      for item in legend_entries:
        leg.plot([], label=mapnames[2] + " " + legendfunction(item),
        markersize=8,
        marker=next(localmarkers),
        color=next(localcolours))
      leg.legend(
        loc = 'lower left',
        ncol=(1,1)[len(legend_entries)>5],
        bbox_to_anchor=(bbox[0],bbox[1]),
        fontsize=8,
        handlelength=3, borderpad=1.2, labelspacing=1.2,
        shadow=True)
      print("added legend at {%i,%i}" % (row, col))
      col += 1
      # if we reach the end of the graph row
      if ((col % numcols)==0):
        col = 0
        row += 1

    plt.tight_layout()
    if options.show_graph :
        plt.show()
    return fig

#----------------------------------------------------------------------------
def insert_safe(a_map, key1, key2, key3, value) :
  #print(key1,key2,key3,value[0],value[1])
  found = False

  # create the 3 level deep map entries if they are not present
  if not (key1) in a_map:
    a_map[key1] = {}
  if not (key2) in a_map[key1]:
    a_map[key1][key2] = {}
  if not (key3) in a_map[key1][key2]:
    a_map[key1][key2][key3] = []

  for item in a_map[key1][key2][key3]:
    if item[0] == value[0]:
      item[1] = item[1]+value[1]
      item[2] += 1
      found = True;
      print(key1,key2,key3,value[0],value[1], "Duplicate", item[2])
      break
  if (not found):
    a_map[key1][key2][key3].append(value + [1])

#----------------------------------------------------------------------------
def average_map(a_map) :
  for key1 in a_map:
    for key2 in a_map[key1]:
      for key3 in a_map[key1][key2]:
        for value in a_map[key1][key2][key3]:
          if value[2]>1:
            value[1] = value[1]/value[2]
