//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TEST_ACTION_MOVE_SEMANTICS_FEB_23_2012_0947AM)
#define HPX_TEST_ACTION_MOVE_SEMANTICS_FEB_23_2012_0947AM

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/access.hpp>

#include <boost/ref.hpp>

namespace hpx { namespace test
{
    // This base class is there to void the is_pod optimization
    // during serialization to make the move semantic tests more meaningful
    struct HPX_COMPONENT_EXPORT object_base
    {
        virtual ~object_base() {};
    };

    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT movable_object
        : object_base
    {
        static std::size_t count;

    public:
        movable_object();

        // Copy constructor.
        movable_object(movable_object const& other);

        // Move constructor.
        movable_object(movable_object && other);

        ~movable_object();

        // Copy assignment.
        movable_object& operator=(movable_object const & other);

        // Move assignment.
        movable_object& operator=(movable_object && other);

        std::size_t get_count() const;
        void reset_count();

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int);

    private:

    };

    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT non_movable_object
        : object_base
    {
        static std::size_t count;

    public:
        non_movable_object();

        // Copy constructor.
        non_movable_object(non_movable_object const& other);

        ~non_movable_object();

        // Copy assignment.
        non_movable_object& operator=(non_movable_object const& other);

        std::size_t get_count() const;
        void reset_count();

        template <typename Archive>
        void load(Archive& ar, const unsigned int);

        template <typename Archive>
        void save(Archive& ar, const unsigned int) const;

        HPX_SERIALIZATION_SPLIT_MEMBER()
    };
}}

#endif

