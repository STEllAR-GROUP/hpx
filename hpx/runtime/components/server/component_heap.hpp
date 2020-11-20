//  Copyright (c) 2011-2017 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/static_reinit/reinitializable_static.hpp>

namespace hpx { namespace components {
    // This is a utility to ensure that there exists exactly one heap
    // per component.
    // This is the customization point and will be defined by the registration
    // macros
    namespace detail {
        template <typename Component>
        struct component_heap_impl;

        template <typename Component>
        typename Component::heap_type&
        component_heap_helper(typename detail::component_heap_impl<Component>::valid*)
        {
            return detail::component_heap_impl<Component>::call();
        }

        template <typename Component> HPX_ALWAYS_EXPORT
        typename Component::heap_type& component_heap_helper(...);
    }

    template <typename Component>
    typename Component::heap_type& component_heap()
    {
        return detail::component_heap_helper<Component>(nullptr);
    }
}}

#define HPX_REGISTER_COMPONENT_HEAP(Component)                                \
namespace hpx { namespace components { namespace detail {                     \
    template <> HPX_ALWAYS_EXPORT                                             \
    Component::heap_type& component_heap_helper< Component>(...)              \
    {                                                                         \
        util::reinitializable_static<Component::heap_type> heap;              \
        return heap.get();                                                    \
    }                                                                         \
}}}                                                                           \
/**/


