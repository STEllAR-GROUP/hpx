//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/serialization/access.hpp>

#include <cstddef>

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

        static std::size_t get_count();
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

        static std::size_t get_count();
        void reset_count();

        template <typename Archive>
        void load(Archive& ar, const unsigned int);

        template <typename Archive>
        void save(Archive& ar, const unsigned int) const;

        HPX_SERIALIZATION_SPLIT_MEMBER()
    };
}}

#endif
