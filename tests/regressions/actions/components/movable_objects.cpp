//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/serialization.hpp>

#include <tests/regressions/actions/components/movable_objects.hpp>

namespace hpx { namespace test
{
    ///////////////////////////////////////////////////////////////////////////
    std::size_t movable_object::count = 0;

    movable_object::movable_object()
    {
        reset_count();
    }

    // Copy constructor.
    movable_object::movable_object(movable_object const& other)
    {
        ++count;
    }

    // Move constructor.
    movable_object::movable_object(BOOST_RV_REF(movable_object) other) {}

    movable_object::~movable_object() {}

    // Copy assignment.
    movable_object& movable_object::operator=(
        BOOST_COPY_ASSIGN_REF(movable_object) other)
    {
        ++count;
        return *this;
    }

    // Move assignment.
    movable_object& movable_object::operator=(BOOST_RV_REF(movable_object) other)
    {
        return *this;
    }

    std::size_t movable_object::get_count() const
    {
        return count;
    }

    void movable_object::reset_count()
    {
        count = 0;
    }

    template <typename Archive>
    void movable_object::serialize(Archive& ar, const unsigned int)
    {
        ar & count;
    }

    template HPX_COMPONENT_EXPORT
    void movable_object::serialize(
        util::portable_binary_oarchive&, const unsigned int);

    template HPX_COMPONENT_EXPORT
    void movable_object::serialize(
        util::portable_binary_iarchive&, const unsigned int);

    ///////////////////////////////////////////////////////////////////////////
    std::size_t non_movable_object::count = 0;

    non_movable_object::non_movable_object()
    {
        reset_count();
    }

    // Copy constructor.
    non_movable_object::non_movable_object(non_movable_object const& other)
    {
        ++count;
    }

    non_movable_object::~non_movable_object() {}

    // Copy assignment.
    non_movable_object& non_movable_object::operator=(non_movable_object const& other)
    {
        ++count;
        return *this;
    }

    std::size_t non_movable_object::get_count() const
    {
        return count;
    }

    void non_movable_object::reset_count()
    {
        count = 0;
    }

    template <typename Archive>
    void non_movable_object::serialize(Archive& ar, const unsigned int)
    {
        ar & count;
    }

    template HPX_COMPONENT_EXPORT
    void non_movable_object::serialize(
        util::portable_binary_oarchive&, const unsigned int);

    template HPX_COMPONENT_EXPORT
    void non_movable_object::serialize(
        util::portable_binary_iarchive&, const unsigned int);
}}


