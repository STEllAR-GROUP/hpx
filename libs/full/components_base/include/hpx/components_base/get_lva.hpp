//  Copyright (c) 2007-2020 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/components_base/traits/is_component.hpp>
#include <hpx/modules/naming_base.hpp>

#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    /// The \a get_lva template is a helper structure allowing to convert a
    /// local virtual address as stored in a local address (returned from
    /// the function \a resolver_client#resolve) to the address of the
    /// component implementing the action.
    ///
    /// The default implementation uses the template argument \a Component
    /// to deduce the type wrapping the component implementing the action. This
    /// is used to get the needed address.
    ///
    /// \tparam Component  This is the type of the component implementing the
    ///                    action to execute.
    template <typename Component, typename Enable = void>
    struct get_lva;

    template <typename Component>
    struct get_lva<Component,
        typename std::enable_if<
            !traits::is_managed_component<Component>::value>::type>
    {
        static Component* call(naming::address_type lva)
        {
            return reinterpret_cast<Component*>(lva);
        }
    };

    template <typename Component>
    struct get_lva<Component,
        typename std::enable_if<
            traits::is_managed_component<Component>::value &&
            !std::is_const<Component>::value>::type>
    {
        static Component* call(naming::address_type lva)
        {
            typedef typename Component::wrapping_type wrapping_type;
            return reinterpret_cast<wrapping_type*>(lva)->get_checked();
        }
    };

    template <typename Component>
    struct get_lva<Component,
        typename std::enable_if<
            traits::is_managed_component<Component>::value &&
            std::is_const<Component>::value>::type>
    {
        static Component* call(naming::address_type lva)
        {
            typedef
                typename std::add_const<typename Component::wrapping_type>::type
                    wrapping_type;
            return reinterpret_cast<wrapping_type*>(lva)->get_checked();
        }
    };
}    // namespace hpx
