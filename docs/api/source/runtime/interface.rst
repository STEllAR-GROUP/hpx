.. _runtime_interface:

*******************
 Runtime Interface 
*******************

.. sectionauthor:: Bryce Lelbach

Synopsis
--------

::

    namespace hpx
    {

        struct runtime
        {
            void start();
            void stop();
        };

        runtime* get_runtime_ptr();
        runtime& get_runtime();
    }

Semantics
---------

.. cpp:class:: hpx::runtime

.. cpp:function:: void hpx::runtime::start()

.. cpp:function:: void hpx::runtime::stop() 

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. cpp:function:: hpx::runtime* get_runtime_ptr()

.. cpp:function:: hpx::runtime& get_runtime()
 
