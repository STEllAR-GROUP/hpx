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

Setup
=====

The source code for this example can be found here:
:download:`hello_world_distributed.cpp
<../../examples/quickstart/custom_serialization.cpp>`.

To compile this program, go to your |hpx| build directory (see
:ref:`hpx_build_system` for information on configuring and building |hpx|) and
enter:

.. code-block:: shell-session

   $ make examples.quickstart.custom_serialization

To run the program type:

.. code-block:: shell-session

   $ ./bin/custom_serialization

This should print:

.. code-block:: text

   Rectangle(Point(x=0,y=0),Point(x=0,y=5))



Member function serialization
-----------------------------

.. literalinclude:: ../../examples/quickstart/custom_serialization.cpp
   :language: c++
   :start-after: //[PointMemberSerialization
   :end-before: //]


HPX is also able to recursively serialize composite classes and structs
given that its members are serializable.

.. literalinclude:: ../../examples/quickstart/custom_serialization.cpp
   :language: c++
   :start-after: //[RectangleMemberSerialization
   :end-before: //]



Free function serialization
-----------------------------

In order to decouple your models from HPX, HPX also allows for the definition
of free function serializers.

.. literalinclude:: ../../examples/quickstart/custom_serialization.cpp
   :language: c++
   :start-after: //[RectangleFREE
   :end-before: //]

Even if you can't modify a class to befriend it, you can still be able to serialize your
class provided that your class is default constructable and you are able to reconstruct it yourself.

.. literalinclude:: ../../examples/quickstart/custom_serialization.cpp
   :language: c++
   :start-after: //[PointClass
   :end-before: //]