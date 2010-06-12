#!/usr/bin/python

import subprocess

from pjplot import Data2D
from pjplot import Gnuplot

from pyrple import Graph
from pyrple import Triple

from pyrple.namespaces import VAR

def query(model, outfilename):
  from namespaces import HPX
  from namespaces import PX

  HPX = HPX()
  PX = PX()

  QueryTriples = [Triple(VAR.thread, HPX.name(), VAR.threadName),
                  Triple(VAR.thread, HPX.numHpxThreads(), VAR.numHpxThreads),
                  Triple(VAR.thread, PX.locality(), VAR.locality),
                  Triple(VAR.locality, HPX.name(), VAR.locality_id)]
  data = []
  for result in model.query(QueryTriples):
    locality = str(result[VAR.locality_id])
    thread_name = str(result[VAR.threadName])
    num_hpx_threads = int(str(result[VAR.numHpxThreads]))

    data.append((locality, thread_name, num_hpx_threads))

  # Write out data to CSV file
  outfile = open(outfilename, 'w')
  for (locality, thread_name, num_hpx_threads) in data:
    outfile.write("'%s', '%s', %d\n" % (locality, thread_name, num_hpx_threads))
  outfile.close()

  # Plot data
  plot_data = Data2D("Task Load Distribution", "OS threads", "HPX Threads")
  c = 1
  max = 0
  for (locality, thread_name, num_hpx_threads) in data:
    x_value = int(locality + thread_name)
    y_value = int(num_hpx_threads)
    plot_data.point(c, y_value)
    c += 1
    if y_value > max: max = y_value
  file = 'plot'
  p = Gnuplot(file)
  p.dataStyle(plot_data, withs='boxes')
  p.xrange(range='0:'+str(c))
  p.yrange(range='0:'+str(max*1.1))
  p.plot()
  p.write()
  subprocess.call(["ps2pdf", file+'.ps', file+'.pdf'])
  subprocess.call(["rm", file+'.ps'])

