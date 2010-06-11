#!/usr/bin/python

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

  outfile = open(outfilename, 'w')
  for result in model.query(QueryTriples):
    locality = str(result[VAR.locality_id])
    thread_name = str(result[VAR.threadName])
    num_hpx_threads = int(str(result[VAR.numHpxThreads]))

    outfile.write("'%s', '%s', %d\n" % (locality, thread_name, num_hpx_threads))
  outfile.close()
