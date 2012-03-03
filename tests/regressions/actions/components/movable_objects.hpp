//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TEST_ACTION_MOVE_SEMANTICS_FEB_23_2012_0947AM)
#define HPX_TEST_ACTION_MOVE_SEMANTICS_FEB_23_2012_0947AM

#include <boost/move/move.hpp>
#include <boost/ref.hpp>

namespace hpx { namespace test
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT movable_object
    {
        static std::size_t count;

    public:
        movable_object();

        // Copy constructor.
        movable_object(movable_object const& other);

        // Move constructor.
        movable_object(BOOST_RV_REF(movable_object) other);

        ~movable_object();

        // Copy assignment.
        movable_object& operator=(BOOST_COPY_ASSIGN_REF(movable_object) other);

        // Move assignment.
        movable_object& operator=(BOOST_RV_REF(movable_object) other);

        std::size_t get_count() const;
        void reset_count();

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int);

    private:
        BOOST_COPYABLE_AND_MOVABLE(movable_object);
    };

    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT non_movable_object
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
        void serialize(Archive& ar, const unsigned int);
    };
}}

#endif

