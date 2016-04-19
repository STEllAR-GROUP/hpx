//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
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
    movable_object::movable_object(movable_object const&)
    {
        ++count;
    }

    // Move constructor.
    movable_object::movable_object(movable_object &&) {}

    movable_object::~movable_object() {}

    // Copy assignment.
    movable_object& movable_object::operator=(
        movable_object const &)
    {
        ++count;
        return *this;
    }

    // Move assignment.
    movable_object& movable_object::operator=(movable_object &&)
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
        serialization::output_archive&, const unsigned int);

    template HPX_COMPONENT_EXPORT
    void movable_object::serialize(
        serialization::input_archive&, const unsigned int);

    ///////////////////////////////////////////////////////////////////////////
    std::size_t non_movable_object::count = 0;

    non_movable_object::non_movable_object()
    {
        reset_count();
    }

    // Copy constructor.
    non_movable_object::non_movable_object(non_movable_object const&)
    {
        ++count;
    }

    non_movable_object::~non_movable_object() {}

    // Copy assignment.
    non_movable_object& non_movable_object::operator=(non_movable_object const&)
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
    void non_movable_object::save(Archive& ar, const unsigned int) const
    {
        ar & count;
    }

    template HPX_COMPONENT_EXPORT
    void non_movable_object::save(
        serialization::output_archive&, const unsigned int) const;

    template <typename Archive>
    void non_movable_object::load(Archive& ar, const unsigned int)
    {
        std::size_t tmp = 0;
        ar & tmp;
        count += tmp;
    }

    template HPX_COMPONENT_EXPORT
    void non_movable_object::load(
        serialization::input_archive&, const unsigned int);
}}


