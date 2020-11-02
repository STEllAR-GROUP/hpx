//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c) 2015-2017 Thomas Heller
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/components_base/traits/component_heap_type.hpp>

#include <cstddef>
#include <new>
#include <utility>

namespace hpx { namespace components
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Component>
        struct simple_heap
        {
            void* alloc(std::size_t count)
            {
                HPX_ASSERT(1 == count);
                return alloc_.allocate(count);
            }
            void free(void* p, std::size_t count)
            {
                HPX_ASSERT(1 == count);
                alloc_.deallocate(static_cast<Component*>(p), count);
            }

            static util::internal_allocator<Component> alloc_;
        };

        template <typename Component>
        util::internal_allocator<Component> simple_heap<Component>::alloc_;
    }
}}

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Enable>
    struct component_heap_type
    {
        using type = hpx::components::detail::simple_heap<Component>;
    };
}}

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    class component : public Component
    {
    public:
        typedef Component type_holder;
        typedef component<Component> component_type;
        typedef component_type derived_type;
        typedef typename traits::component_heap_type<Component>::type heap_type;

        /// \brief Construct a simple_component instance holding a new wrapped
        ///        instance
        template <typename... Ts>
        component(Ts&&... vs)
          : Component(std::forward<Ts>(vs)...)
        {}
    };
}}

