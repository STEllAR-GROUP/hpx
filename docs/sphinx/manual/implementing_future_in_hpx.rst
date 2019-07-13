..
    Copyright (C) 2019 Tapasweni Pathak

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _examples_future:

==========================================
Object Representing A Result: Future
==========================================

A future is an object representing a result which has not been calculated yet

.. figure:: /_static/images/future.png

- Enables transparent synchronization with producer
- Hides notion of dealing with threads
- Makes asynchrony manageable
- Allows for composition of several asynchronous operations
- Turns concurrency into parallelism

Future API
==========

.. code-block:: cpp

   template <typename R>
   class future
   {
      // future constructors
      // Query the state
      // Waiting on the result
   };

   template <typename R>
   class shared_future
   {
      // Future constructors
      // Query the state
      // Waiting on the result
   };


Constructing a ``hpx::future<R>``

.. code-block:: cpp

   template <typename R>
   class future
   {
      // Future constructors
      // Construct an empty future.
      future();

      // Move a future to a new one
      future(future<R>&& f);

      // Unwrap a future. The new future becomes ready when
      // the inner, and outer futures are ready.
      explicit future(future<future<R>>&& f);
      explicit future(future<shared_future<R>>&& f);

      // Turn this future into a shared_future. Invalidates the future!
      shared_future<R> share();

      // Query the state
      // Waiting on the result
    };


Querying the state of the future:

.. code-block:: cpp

    template <typename R>
    class future
    {
        // Future constructors

    		// Query the state

    		// Check if the future is ready yet.
    		bool is_ready();

    		// Check if the future has a value
    		bool has_value();

    		// Check if the future has an exception
    		bool has_exception();

    		// Waiting on the result
		};

Waiting for the future to become ready:

.. code-block:: cpp

    template <typename R>
		class future
		{
    // Future constructors, Query the state...

    // Waiting on the result
    void wait() const;

    // Waiting for the result, but not longer than until given time point
    template <typename Clock, typename Duriation>
    future_status wait_until(
        std::chrono::time_point<Clock, Duration> const& abs_time) const;

    // Waiting for the result, but not longer than give duration
    template <typename Rep, typename Period>
    future_status wait_for(
        std::chrono::duration<Rep, Period> const& rel_time) const;

    // Get the result...
    };


Constructing a hpx::shared_future<R>:

.. code-block:: cpp

    template <typename R>
    class shared_future
    {
      // Future constructors
      // Construct an empty future.
      shared_future();

      // Move a future to a new one
      shared_future(shared_future<R>&& f);

      // Share ownership between two futures
      shared_future(shared_future<R> const& f);

      // Unwrap a future. The new future becomes ready when
      // the inner, and outer future are ready.
      explicit shared_future(shared_future<future<R>>&& f);

      // implicitly share a future
      shared_future(future<R>&& f);

      // Query the state
      // Waiting on the result
    };

Waiting for the future to become ready:


.. code-block:: cpp

   template <typename R>
   class shared_future
   {
      // Future constructors
      // Query the state

      // Waiting on the result
      void wait() const;

      // Get the result. This function might block if the result has
      // not been computed yet.
      R const& get();

      // Attach a continuation. The function f gets called with
      // the (ready) future. Returns a new future with the result of
      // the invocation of f.
      template <typename F>
      auto then(F&& f) const;
   };

Producing Futures
=================

`hpx::async`

.. code-block:: cpp

     template <typename F, typename... Ts>
     auto async(F&& f, Ts&&... ts)
    -> future<decltype(f(std::forward<Ts>(ts)...)>;

``F`` is anything callable with the passed arguments (actions are callable)

.. code-block:: cpp

      template <typename F, typename... Ts>
      auto async(launch_policy, F&& f, Ts&&... ts)
     -> future<decltype(f(std::forward<Ts>(ts)...)>;

``launch_policy`` can be ``async``, ``sync``, ``fork``, ``deferred``

.. code-block:: cpp

      template <typename Executor typename F, typename... Ts>
      auto async(Executor&&, F&& f, Ts&&...  ts)
     -> future<decltpype(f(std::forward<Ts>(ts)...)>;

``hpx::lcos::local::promise``

.. code-block:: cpp

    hpx::lcos::local::promise<int> p;       // local only
    hpx::future<int> f = p.get_future();
    // f.is_ready() == false, f.get(); would lead to a deadlock

    p.set_value(42);

    // Print 42
    std::cout << f.get() << std::endl;

``hpx::promise``

.. code-block:: cpp

     hpx::promise<int> p;                    // globally visible
     hpx::future<int> f = p.get_future();
     // f.is_ready() == false, f.get(); would lead to a deadlock

     hpx::async(
        [](hpx::id_type promise_id)
        { 
        hpx::set_lco_value(promise_id, 42);
        }
        , p.get_id());

    // Print 42
    std::cout << f.get() << std::endl;

``hpx::make_ready_future``

.. code-block:: cpp

     template <typename T>
     future<typename decay<T>::type> make_ready_future(T&& t);

     future<void> make_ready_future();


Composing Futures
=================

Sequential Composition: ``future::then``

.. code-block:: cpp

    future<int> f1 = hpx::async(...);

    // Call continuation once f1 is ready. f2 will become ready once
    // the continuation has been run.
    future<double> f2 = f1.then(
        [](future<int>&& f) { return f.get() + 0.0; });

- The continuation needs to take the future as parameter to allow for exception
handling. Exceptions happening in asynchronous calls will get rethrown on `.get()`
- then accepts launch policies as well as executors
- `f1` will get invalidated.

No invalidation:

.. code-block:: cpp

    shared_future<int> f1 = hpx::async(...);

    // Call continuation once f1 is ready. f2 will become ready once
    // the continuation has been run.
    future<double> f2 = f1.then(
        [](future<int>&& f) { return f.get() + 0.0; });

Or Composition: ``when_any``

.. code-block:: cpp

    std::vector<future<int>> fs = ...;

    future<int> fi =
    hpx::when_any(fs).then(
        [](auto f)
        {
            auto res = f.get();
            return res.futures[res.index];
        });

- Allows for waiting on any of the input futures
- Returns a `future<when_any_result<Sequence>>`:

.. code-block:: cpp

     template <typename Sequence>
     struct when_any_result
     {
       std::size_t index; // Index to a future that became ready
       Sequence futures;  // Sequence of futures
     };

