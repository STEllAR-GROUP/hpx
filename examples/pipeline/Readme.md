<!-- Copyright (c) 2018 Thomas Heller                                             -->
<!--                                                                              -->
<!-- SPDX-License-Identifier: BSL-1.0                                             -->
<!-- Distributed under the Boost Software License, Version 1.0. (See accompanying -->
<!-- file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)        -->

This example demonstrates a pipeline split up in 3 processes:
 - emitter.cpp:
    * Process producing input for worker. This serves as the master process
 - worker.cpp:
    * Process working on inputs. There can be multiple instances of this process
 - collector.cpp:
    * Process collecting the result of the workers

When using the MPI parcelport, the example can be run like this:

```
mpirun -np 1 ./bin/emitter : -np 1 ./bin/collector : -np N-1 ./bin/worker
```

For elasticity, the applicate can be started as following:

```
host0$ ./bin/emitter --hpx:hpx=<host0> --hpx:console
host1$ ./bin/collector --hpx:hpx=<host1> --hpx:agas=<host0> --hpx:connect --hpx:run-hpx-main
host2$ ./bin/worker --hpx:hpx=<host2> --hpx:agas=<host0> --hpx:connect --hpx:run-hpx-main
host3$ ./bin/worker --hpx:hpx=<host3> --hpx:agas=<host0> --hpx:connect --hpx:run-hpx-main
...
hostN$ ./bin/worker --hpx:hpx=<hostN> --hpx:agas=<host0> --hpx:connect --hpx:run-hpx-main
```
