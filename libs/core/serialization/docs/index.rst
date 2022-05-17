..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_serialization:

=============
serialization
=============

This module provides serialization primitives and support for all built-in
types as well as all C++ Standard Library collection and utility types. This
list is extended by |hpx| vocabulary types with proper support for global
reference counting. |hpx|'s mode of serialization is derived from `Boost's
serialization model
<https://www.boost.org/doc/libs/1_72_0/libs/serialization/doc/index.html>`_
and, as such, is mostly interface compatible with
its Boost counterpart.

The purest form of serializing data is to copy the content of the payload bit
by bit; however, this method is impractical for generic C++ types, which might
be composed of more than just regular built-in types. Instead, |hpx|'s approach
to serialization is derived from the Boost Serialization library, and is geared
towards allowing the programmer of a given class explicit control and syntax of
what to serialize. It is based on operator overloading of two special archive
types that hold a buffer or stream to store the serialized data and is
responsible for dispatching the serialization mechanism to the intrusive or
non-intrusive version. The serialization process is recursive. Each member that
needs to be serialized must be specified explicitly. The advantage of this
approach is that the serialization code is written in C++ and leverages all
necessary programming techniques. The generic, user-facing interface allows
for effective application of the serialization process without obstructing the
algorithms that need special code for packing and unpacking. It also allows for
optimizations in the implementation of the archives.

See the :ref:`API reference <modules_serialization_api>` of the module for more
details.


=============
Serializing user-defined types
=============

Just like boost, hpx allows users to serialize user-defined types by either
providing the serializer as a member function or defining the serialization as a
free function.

+++++++++
Member function serialization
+++++++++

.. code-block:: c++

    #include <hpx/serialization.hpp>

    class PointClass { // Member function example
        int x{0};
        int y{0};


    public:
        PointClass(int x, int y) : x(x), y(y) {}

        PointClass() = default; // Required to instantiate the reconstruction on the receiving node

        [[nodiscard]] int getX() const noexcept {
            return x;
        }

        [[nodiscard]] int getY() const noexcept {
            return y;
        }

    private:
        friend class hpx::serialization::access; // Allows HPX to access private members

        template<typename Archive>
        void serialize(Archive &ar, const unsigned int version) {
            ar & x & y;
        }
    };

HPX is also able to recursively serialize composite classes and structs
given that the members are serializable.

.. code-block:: c++

    #include <hpx/serialization.hpp>

    struct Rectangle {
        PointClass top_left;
        PointClass lower_right;

        template<typename Archive>
        void serialize(Archive &ar, const unsigned int version) {
            ar & top_left & lower_right;
        }

    };

+++++++++
Free function serialization
+++++++++

In order to decouple your models from HPX, HPX also allows for the definition
of free function serializers.

.. code-block:: c++

    #include <hpx/serialization.hpp>

    struct Rectangle {
        PointClass top_left;
        PointClass lower_right;
    };

    template <typename Archive>
    void serialize(Archive &ar, Rectangle& pt, const unsigned int version){
        ar & pt.lower_right & pt.top_left;
    }

Even if you can't modify a class to befriend it, hpx might still be able to serialize your
class provided.If your class provides an assignment operator and your class
is default constructable.

.. code-block:: c++

    #include <hpx/serialization.hpp>

    class PointClass {

    public:
        PointClass(int x, int y) : x(x), y(y) {}

        PointClass() =default;

        [[nodiscard]] int getX() const noexcept {
            return x;
        }


        [[nodiscard]] int getY() const noexcept {
            return y;
        }

    private:
        int x;
        int y;
    };

    template <typename Archive>
    void load(Archive &ar, PointClass& pt, const unsigned int version)
    {
        int x, y;
        ar >> x >> y;
        pt = PointClass(x, y);
    }

    template <typename Archive>
    void save(Archive &ar, PointClass const& pt, const unsigned int version)
    {
        ar << pt.getX() << pt.getY();
    }

    HPX_SERIALIZATION_SPLIT_FREE(PointClass); // Required