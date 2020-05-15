HEP 2: The public API for |hpx| 2.0.0
=====================================

Motivation
----------

|hpx| has evolved to have a rather ad-hoc organization of functionality into
namespaces, headers, and libraries. This HEP specifies the public API, and
through which namespaces and headers the public API is accessible, for |hpx|
2.0.0.

Implementation
--------------

The public API is presented as

- ``header name``
  - ``hpx::x``
  - ``hpx::y``

where including ``header name`` gives you access to at least ``hpx::x`` and
``hpx::y``. Functionality may be organized differently internally (both headers
and namespaces).

Functionality that corresponds to, or extends, standard library functionality,
will be in the same namespace as in the standard library with ``std`` replaced
by ``hpx``, and a header ``abc`` replaced by ``hpx/abc.hpp``. Functionality that
is proposed for the standard library, and may thus be in ``std::experimental``
and a header ``experimental/abc`` does not need to be in the
``hpx::experimental`` namespace or a ``hpx/experimental`` header if the
functionality is considered stable in |hpx|. Distributed extensions to local
parallelism and concurrency will be in the ``hpx::distributed`` namespace and
``hpx/distributed`` headers.

Utilities
.........

- ``hpx/functional.hpp``
  - ``hpx::function``
  - ``hpx::unique_function``
  - ``hpx::function_nonser``
  - ``hpx::unique_function_nonser``
  - ``hpx::bind``
  - ``hpx::bind_front``
  - ``hpx::bind_back``
  - ``hpx::invoke``
  - ``hpx::invoke_fused``
  - ``hpx::result_of``
  - more?

- ``hpx/tuple.hpp``
  - ``hpx::tuple``

- ``hpx/any.hpp``
  - ``hpx::any``

- ``hpx/optional.hpp``
  - ``hpx::optional``

- ``hpx/format.hpp``
  - ``hpx::format``?
  - other utilities that go with ``format``

- ``hpx/chrono.hpp``
  - ``hpx::high_resolution_clock``?
  - ``hpx::high_resolution_timer``?
  - ``hpx::scoped_timer``?

- ``hpx/exception.hpp``
  - ``hpx::exception``
  - ``HPX_THROW_EXCEPTION``

- ``hpx/system_error.hpp``
  - ``hpx::error_code``

- ``hpx/assert.hpp``
  - ``HPX_ASSERT``

- ``hpx/serialization.hpp``
  - what functionality actually needs to be public?
  - Some classes have intrusive serialization in which case there's no need to
    include ``serialization.hpp`` to get the serialization support; others have
    it implemented in the serialization module. Should we just state that using
    serialization for any class requires including ``hpx/serialization.hpp``
    even though it may already be included transitively?

Local concurrency and parallelism
.................................

- ``hpx/future.hpp``
  - ``hpx::future``
  - ``hpx::promise``
  - ``hpx::shared_future``
  - ``hpx::async``
  - ``hpx::sync``
  - ``hpx::apply``
  - ``hpx::dataflow``
  - ``hpx::launch_policy`` (or ``hpx::execution::launch_policy``?)
  - ``hpx::is_future`` and friends

- ``hpx/future.hpp``?
  - ``hpx::when_all``
  - ``hpx::wait_all``
  - ``hpx::when_any``
  - ``hpx::wait_any``
  - ``hpx::when_some``
  - ``hpx::wait_some``
  - ``hpx::when_each``
  - ``hpx::wait_each``
  - ``hpx::split_future``

- ``hpx/thread.hpp``
  - ``hpx::thread``
  - ``hpx::this_thread::yield``
  - ``hpx::this_thread::get_id``
  - ``hpx::this_thread::sleep_for``
  - ``hpx::this_thread::sleep_until``

- ``hpx/algorithm.hpp``
  - ``hpx::for_loop`` and all the other parallel algorithms

- ``hpx/algorithm/for_loop.hpp``
  - ``hpx::for_loop`` and the same for all other individual algorithms (no need to separate range-based and iterator-based, I believe)

- ``hpx/execution.hpp``
  - ``hpx::execution::sequenced_policy``
  - ``hpx::execution::unsequenced_policy``
  - ``hpx::execution::parallel_sequenced_policy``
  - ``hpx::execution::parallel_unsequenced_policy``
  - ``hpx::execution::sequenced_task_policy``
  - ``hpx::execution::parallel_task_policy``
  - ``hpx::execution::seq``
  - ``hpx::execution::unseq`` (or is it ``datapar``...?)
  - ``hpx::execution::par``
  - ``hpx::execution::par_unseq``
  - ``hpx::execution::task``

- ``hpx/execution.hpp``? these should probably be public, despite upcoming changes to executors?
  - ``hpx::execution::post``?
  - ``hpx::execution::sync_execute``?
  - ``hpx::execution::async_execute``?
  - ``hpx::execution::then_execute``?
  - ``hpx::execution::bulk_sync_execute``?
  - ``hpx::execution::bulk_async_execute``?
  - ``hpx::execution::bulk_then_execute``?

- ``hpx/execution.hpp``?
  - ``hpx::execution::sequenced_executor``
  - ``hpx::execution::parallel_executor``
  - all the other executors? or just a selected subset?
  - ``hpx::execution::service_executor``
  - ``hpx::experimental::execution::thread_pool_executor``?
  - ``hpx::execution::current_executor`` (not sure we need this)
  - ``hpx::execution::is_executor`` and friends

- ``hpx/execution.hpp``
  - ``hpx::execution::static_chunk_size`` and other executor parameters

- ``hpx/mutex.hpp``
  - ``hpx::mutex``
  - ``hpx::no_mutex``
  - ``hpx::recursive_mutex``
  - ``hpx::shared_mutex``
  - ``hpx::spinlock``
  - ``hpx::call_once``

- ``hpx/condition_variable.hpp``
  - ``hpx::condition_variable``

- ``hpx/semaphore.hpp``
  - ``hpx::counting_semaphore``
  - ``hpx::sliding_semaphore``

- ``hpx/barrier.hpp``
  - ``hpx::barrier``

- ``hpx/latch.hpp``
  - ``hpx::latch``

- ``hpx/channel.hpp``?
  - ``hpx::channel`` (``hpx::channel_*``?)

- ``hpx/spmd_block.hpp``?
  - ``hpx::spmd_block``

- ``hpx/task_block.hpp``?
  - ``hpx::task_block``

Distributed
...........

- ``hpx/distributed/future.hpp``?
  - distributed equivalents of ``hpx::promise``, ``hpx::async``, etc.

- ``hpx/distributed/algorithm.hpp``?
  - distributed equivalents of parallel algorithms

- ``hpx/distributed/latch.hpp``
  - ``hpx::distributed::latch``
- ``hpx/distributed/barrier.hpp``
  - ``hpx::distributed::barrier``
- etc. to match local headers and namespaces

- ``hpx/distributed/components.hpp``?
  - ``HPX_REGISTER_COMPONENT``
  - any other classes, functions, and macros that might be required to write components?

- ``hpx/distributed/actions.hpp``?
  - ``HPX_REGISTER_ACTION``
  - any other classes, functions, and macros that might be required to write actions?

- ``hpx/distributed/channel.hpp``?
  - ``hpx::distributed::channel``

- ``???`` each in a separate header like channel, latch, etc.?
  - ``hpx::distributed::all_to_all``
  - ``hpx::distributed::all_reduce``
  - ``hpx::distributed::broadcast``
  - ``hpx::distributed::fold``
  - ``hpx::distributed::gather``
  - ``hpx::distributed::reduce``
  - ``hpx::distributed::spmd_block``

- ``hpx/distributed/checkpoint.hpp``?
  - ``hpx::distributed::checkpoint`` stable?
  - this is not necessarily distributed only...

- ``hpx/experimental/distributed/resiliency.hpp``?
  - ``hpx::experimental::distributed::async_replay`` and other resiliency?
  - ready for public use or still experimental?
  - the standard libraries use ``std::experimental::x``, perhaps we should just
    follow that instead? the argument for ``hpx::x::experimental`` wasn't that
    strong anyway...
  - this is not necessarily distributed only...

- ``hpx/distributed.hpp``
  - all above headers

Runtime
.......

- ``hpx/runtime.hpp``?
  - ``hpx::get_os_thread_count``
  - ``hpx::get_worker_thread_num``
  - ``hpx::get_thread_name``
  - other ``hpx::get_*`` functions
  - ``hpx::register_thread``
  - ``hpx::unregister_thread``
  - ``hpx::register_startup_function``
  - ``hpx::register_shutdown_function``
  - ``hpx::is_*``
  - ``hpx::state`` (rename to ``hpx::runtime_state`` for the runtime?)
  - ``hpx::get_locality_id`` this and the following are not quite right here,
    but they can return reasonable defaults even in the local case...
  - ``hpx::get_locality_name``
  - ``hpx::get_num_localities``
  - note: ``hpx::runtime`` does not need to be public; the above global
    functions should be enough for interacting with the runtime. They also allow
    us to have checks that the runtime is created or running (as we already do).

- ``hpx/distributed/runtime.hpp``?
  - all of the above and (should they be in ``hpx::distributed`` or not?)
  - ``hpx::find_localities``
  - ``hpx::find_here``

- ``hpx/init.hpp``?
  - ``hpx::init`` (rename to ``hpx::start_stop`` or something similar? only
    leave ``hpx::start`` and ``hpx::stop``? I found the relationship between
    ``start``, ``stop``, ``init``, and ``finalize`` confusing at least in the
    beginning.)
  - ``hpx::start``
  - ``hpx::stop``
  - ``hpx::finalize``
  - ``hpx::resume``?
  - ``hpx::suspend``?
  - ``hpx::init_parameters``

- performance counters?

- ``hpx/wrap_main.hpp`` (I propose to rename ``hpx_main.hpp`` to
  ``wrap_main.hpp`` to avoid confusion with the implicit entry point used by
  ``hpx::init`` and ``hpx::start`` being ``hpx_main``)

- There is a lot of scheduler and thread pool functionality that we could make
  public, but I think they're not stable (or conformant?) enough to be
  considered for the public API. Users may still use them but we don't give the
  same stability guarantees as for the public API. We could consider putting
  ``scheduler_base`` and ``thread_pool_base`` in ``hpx::experimental::`` and
  ``hpx/experimental/execution.hpp`` as they should eventually be stable?
  - ``hpx::experimental::get_pool``
  - ``hpx::experimental::partitioner``

Other
.....

- ``hpx/version.hpp``
  - ``hpx::major_version`` and all the other version functions
  - ``HPX_VERSION_MAJOR`` and all the other version macros

|cmake| targets
...............

TODO

- ``HPX::hpx_core``
- ``HPX::hpx_local``
- ``HPX::hpx`` (does not wrap main automatically)
- ``HPX::wrap_main``
- ``HPX::x_component``
- ...
