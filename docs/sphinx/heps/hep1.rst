HEP 1: |hpx| build configurations and exported libraries
========================================================

Motivation
----------

The motivation for this change is to expose smaller parts of |hpx| for users to
consume. This is meant to address both the case where a user builds |hpx|
manually and and wishes to build a subset of |hpx|, and the case where a user
uses a preinstalled |hpx| (distribution packages, cluster modules) which may
have most features enabled but still only wishes to pull in a subset of |hpx|.
For the latter the user may choose to include headers and link to only a subset
of |hpx| or they may choose to start only the local runtime even though |hpx|
was built with the distributed runtime.

Implementation
--------------

|hpx| will provide the following targets and corresponding libraries:

- ``HPX::hpx_core``: Thread pools, schedulers, coroutines, and utilities;
  contains functionality that is essential for building |hpx| but not the main
  user-facing functionality itself (e.g. ``hpx::thread``, ``hpx::future``).
- ``HPX::hpx_local``: Depends on ``HPX::hpx_core``; contains functionality
  corresponding to the standard libraries for concurrency and parallelism, and
  local extensions (e.g. ``hpx::thread``, ``hpx::async``, parallel algorithms,
  synchronization primitives meant for use on ``hpx::thread``\ s).
- ``HPX::hpx``: Depends on everything above; contains the local and distributed
  runtimes, distributed extensions to the local library for concurrency and
  parallelism, and utilities and functionality that depend on the runtime.

Including headers from ``HPX::hpx_local`` will not include headers from
``HPX::hpx``. The granularity at which features can be disabled at compile time
in |hpx| are the targets in this HEP because it is a manageably small set of
targets, unlike the fine-grained modules used for organizing the library
internally. The smallest configuration of |hpx| that can be enabled is
``HPX::hpx_core``. The fine-grained modules will be exposed as static libraries
for users wishing to link to an even smaller part of |hpx|.

The ``HPX::hpx`` library will expose the runtime initialization functionality
(``hpx::init`` and ``hpx::start``). Exposing the local and distributed runtimes
from one library means that one can dispatch to the two runtimes at run time.
|hpx| will expose an additional option for initializing the runtime:
``runtime_type``. This can be ``default``, ``local``, or ``distributed``.
Including ``hpx_init.hpp`` or ``hpx_start.hpp`` will not pull in headers related
to distributed functionality. Even if |hpx| was built with the distributed
runtime the user can choose to start the local runtime and not pull in any
distributed headers. If the user sets the runtime type to ``local`` the local
runtime will always be started. If the user sets the runtime type to
``distributed`` the distributed runtime will be started if it was enabled.
Otherwise it will cause a run time error. If the runtime type is ``default`` the
most featureful available runtime is started, i.e. the distributed runtime if
available and the local runtime otherwise. The behavior for existing users of
distributed features who do not explicitly specify the runtime type will be
unchanged.
