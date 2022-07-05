..
    Copyright (C) 2022 John Sorial

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _examples_serialization:


==============================
Serializing user-defined types
==============================

In order to facilitate the sending and receiving of complex datatypes HPX provides a
serialization abstraction.

Just like boost, hpx allows users to serialize user-defined types by either
providing the serializer as a member function or defining the serialization as a
free function.

**Unlike Boost HPX doesn't acknowledge second unsigned int parameter, it is solely there to preserve
API compatibility with Boost Serialization**

This is tutorial was heavily inspired by
`Boost's serialization concepts <https://www.boost.org/doc/libs/1_79_0/libs/serialization/doc/serialization.html>`_.

Setup
=====

The source code for this example can be found here:
:download:`custom_serialization.cpp
<../../examples/quickstart/custom_serialization.cpp>`.

To compile this program, go to your |hpx| build directory (see
:ref:`building_hpx` for information on configuring and building |hpx|) and
enter:

.. code-block:: shell-session

   $ make examples.quickstart.custom_serialization

To run the program type:

.. code-block:: shell-session

   $ ./bin/custom_serialization

This should print:

.. code-block:: text

   Rectangle(Point(x=0,y=0),Point(x=0,y=5))
   gravity.g = 9.81%


Serialization Requirements
===========================
In order to serialize objects in HPX, at least one of the following criteria must be met:

In the case of default constructible objects:

* The object is an empty type.
* Has a serialization function as shown in this tutorial.
* All members are accessible publicly and they can be used in structured binding contexts.

Otherwise:

* They need to have  :ref:`special serialization support <Serializing non default constructable classes>`.

Member function serialization
-----------------------------

.. literalinclude:: ../../examples/quickstart/custom_serialization.cpp
   :language: c++
   :start-after: //[point_member_serialization
   :end-before: //]

Notice that ``point_member_serialization`` is defined as bitwise serializable
(see :ref:`bitwise_serialization` for more details).
HPX is also able to recursively serialize composite classes and structs
given that its members are serializable.

.. literalinclude:: ../../examples/quickstart/custom_serialization.cpp
   :language: c++
   :start-after: //[rectangle_member_serialization
   :end-before: //]



Free function serialization
-----------------------------

In order to decouple your models from HPX, HPX also allows for the definition
of free function serializers.

.. literalinclude:: ../../examples/quickstart/custom_serialization.cpp
   :language: c++
   :start-after: //[rectangle_free
   :end-before: //]

Even if you can't modify a class to befriend it, you can still be able to
serialize your class provided that your class is default constructable
and you are able to reconstruct it yourself.

.. literalinclude:: ../../examples/quickstart/custom_serialization.cpp
   :language: c++
   :start-after: //[point_class
   :end-before: //]


Serializing non default constructable classes
----------------------------------------------
.. _non-default_constructable

Some classes don't provide any default constructor.

.. literalinclude:: ../../examples/quickstart/custom_serialization.cpp
   :language: c++
   :start-after: //[planet_weight_calculator
   :end-before: //]

In this case you have to define a ``save_construct_data`` and ``load_construct_data`` in which you
do the serialization yourself.

.. literalinclude:: ../../examples/quickstart/custom_serialization.cpp
   :language: c++
   :start-after: //[save_construct_data
   :end-before: //]

.. _bitwise_serialization:

Bitwise serialization for bitwise copyable data
-----------------------------------------------

When sending non arithmetic types not defined by
`std::is_arithmetic <https://en.cppreference.com/w/cpp/types/is_arithmetic>`_, HPX has to (de)serialize each object
separately. However, if the class you are trying to send classes consists only of bitwise copyable datatypes,
you may mark your class as such.
Then HPX will serialize your object bitwise instead of element wise.
This has enormous benefits, especially when sending a vector/array of your class.
To define your class as such you need to call ``HPX_IS_BITWISE_SERIALIZABLE(T)`` with your desired custom class.

.. literalinclude:: ../../examples/quickstart/custom_serialization.cpp
   :language: c++
   :start-after: //[point_member_serialization
   :end-before: //]
