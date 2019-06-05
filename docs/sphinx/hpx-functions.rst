.. _HPX Functions:

=============
HPX Functions
=============

This document describes the prototypes, the allowable arguments of the listed
HPX functions.

async
=====

Async in C++ allows running asynchronous programming. Async programming is useful
in compute heavy implementations. Basic async function would look like

.. code-block:: text
   #include <hpx/lcos/future.hpp>
   using namespace hpx;
   int main() {
    future<int> f = async([]() -> int { return 91; });
    int val = f.get();
   }

You can chain multiple async operations. For implementing chain async programming
``objFuture.then(...).then(...).then(...).then(...).then(...)``.

Implementing async functions should have input parameters as

- Function/Function Object/Lambda
- Executor function
- Launch policy

then
====

then in asychronous programming is used in chained async implementations. Using
then function will avoid blocking waits or wasting threads on polling. Future
can have then chained implementations. With then the antecedent future is ready
(has a value or exception stored) before the continuation starts as instructed
by lambda function. Example implementation using hpx is

.. code-block:: text

   #include <hpx/lcos/future.hpp>

   int main() {
     future<int> f = async([]() -> int { return 91; });
     future<string> fB = fA.then([](future<int> f)) {
       return f.get().to_string();
     });
   }


wait
====

``hpx::future<T>::wait`` blocks the function implementation until the call is
available. This function call can be checked using ``valid() == true``. Example
implementation using hpx is

.. code-block:: text

   #include <hpx/lcos/future.hpp>
   #include <thread>

   int main(){
     hpx::future<int> fA = hpx::async([])(){});{
       func(valA);
     });
     hpx::future<string> fB = hpx::async(hpx::launch::async, [](){
       func(valB);
     });

     fA.wait();
     fB.wait();

     hpx::cout << fA.get() << `\n`;
     hpx::cout << fB.get() << `\n`;
   }

wait_all
========

The function ``wait_all`` is an operator allowing to join on the result of all
given futures. It AND-composes all future objects given and returns after they
finished executing. The function ``wait_all`` returns after all futures have
become ready. All input futures are still valid after wait all returns.

.. literalinclude:: ../../hpx/lcos/wait_all.hpp
   :lines: 48-91

.. code-block:: text

   #include <hpx/lcos/future.hpp>
   int main() {
     std::vector<hpx::future<void>> results;
     for (int i = 0; i != NUM; ++i)
       results.push_back(hpx::async(...));
       hpx::wait_all(results);
     }
   }
