//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/libs/iostreams for documentation.
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

// Contains the definition of the class template access_control, which
// allows the type of inheritance from a provided base class to be specified
// using a template parameter.

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/type_support.hpp>

#include <type_traits>

namespace hpx::iostreams {

    struct protected_t    // Represents protected inheritance.
    {
    };

    struct public_t    // Represents public inheritance.
    {
    };

    namespace detail {

        // Implements protected inheritance.
        template <typename U>
        struct prot_t : protected U
        {
            using U::U;
        };

        // Implements public inheritance.
        template <typename U>
        struct pub_t : public U
        {
            using U::U;
        };

        //
        // Used to deduce the base type for the template access_control.
        //
        template <typename T, typename Access>
        struct access_control_base
        {
            using bad_access_specifier = int;

            // clang-format off
            using type = util::select_t<
                std::is_same<Access, protected_t>, prot_t<T>, 
                std::is_same<Access, public_t>, pub_t<T>,
                util::else_t, bad_access_specifier>;
            // clang-format on
        };
    }    // namespace detail

    //
    // Template name: access_control.
    // Description: Allows the type of inheritance from a provided base class
    //      to be specified using an int template parameter.
    // Template parameters:
    //      Base - The class from which to inherit (indirectly.)
    //      Access - The type of access desired. Must be one of the
    //          values access_base::prot or access_base::pub.
    //
    template <typename T, typename Access>
    struct access_control : public detail::access_control_base<T, Access>::type
    {
        using base_type = detail::access_control_base<T, Access>::type;
        using base_type::base_type;
    };
}    // namespace hpx::iostreams
