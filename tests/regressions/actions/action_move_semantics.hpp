//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TEST_ACTION_MOVE_SEMANTICS_FEB_23_2012_0947AM)
#define HPX_TEST_ACTION_MOVE_SEMANTICS_FEB_23_2012_0947AM

#include <boost/move/move.hpp>

namespace hpx { namespace test
{
    ///////////////////////////////////////////////////////////////////////////
    struct movable_object
    {
        movable_object() : copy_count(0) {}

        // Copy constructor.
        movable_object(movable_object const& other)
          : copy_count(other.copy_count + 1)
        {}

        // Move constructor.
        movable_object(BOOST_RV_REF(movable_object) other)
          : copy_count(other.copy_count)
        {}

        ~movable_object() {}

        // Copy assignment.
        movable_object& operator=(BOOST_COPY_ASSIGN_REF(movable_object) other)
        {
            copy_count = other.copy_count + 1;
            return *this;
        }

        // Move assignment.
        movable_object& operator=(BOOST_RV_REF(movable_object) other)
        {
            copy_count = other.copy_count;
            return *this;
        }

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & copy_count;
        }

        std::size_t copy_count;

    private:
        BOOST_COPYABLE_AND_MOVABLE(movable_object);
    };

    ///////////////////////////////////////////////////////////////////////////
    struct non_movable_object
    {
        non_movable_object() : copy_count(0) {}

        // Copy constructor.
        non_movable_object(non_movable_object const& other)
          : copy_count(other.copy_count + 1)
        {}

        ~non_movable_object() {}

        // Copy assignment.
        non_movable_object& operator=(non_movable_object const& other)
        {
            copy_count = other.copy_count + 1;
            return *this;
        }

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & copy_count;
        }

        std::size_t copy_count;
    };
}}

#endif

