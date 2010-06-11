#!/usr/bin/python

from pyrple import Graph
from pyrple import Triple

from pyrple.namespaces import VAR


def query(model, outfilename):
  from namespaces import HPX

  HPX = HPX()

  QueryTriples = [Triple(VAR.thread, HPX.name(), VAR.threadName),
                  Triple(VAR.thread, HPX.numHpxThreads(), VAR.numHpxThreads)]

  outfile = open(outfilename, 'w')
  for result in model.query(QueryTriples):
    thread_name = str(result[VAR.threadName])
    num_hpx_threads = int(str(result[VAR.numHpxThreads]))

    outfile.write("'%s', %d\n" % (thread_name, num_hpx_threads))
  outfile.close()
