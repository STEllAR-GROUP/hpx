..
    Copyright (C) 2019 Tapasweni Pathak

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _examples_components:

========================================================
Writing Components Enabling Gloabl Object: components
=========================================================


Implementing global object using component
==========================================

.. code-block:: cpp

   struct hello_world_component;
   struct hello_world;

   int main()
   {
      hello_world hw(hpx::find_here());

      hw.print();
   }


.. code-block:: cpp

    struct hello_world_component;

    // Client implementation
    struct hello_world
    : hpx::components::client_base<hello_world, hello_world_component>
    {
       hello_world(hpx::id_type where);
       hpx::future<void> print();
    };

    int main()
    {
       hello_world hw(hpx::find_here());
       hw.print();
    }
