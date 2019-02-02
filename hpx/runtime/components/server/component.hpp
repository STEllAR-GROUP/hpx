//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c) 2015-2017 Thomas Heller
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_COMPONENTS_SERVER_COMPONENT_HPP
#define HPX_RUNTIME_COMPONENTS_SERVER_COMPONENT_HPP

#include <hpx/config.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/internal_allocator.hpp>

#include <cstddef>
#include <new>
#include <utility>

namespace hpx { namespace components {

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

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    class component : public Component
    {
    public:
        typedef Component type_holder;
        typedef component<Component> component_type;
        typedef component_type derived_type;
        typedef detail::simple_heap<Component> heap_type;

        /// \brief Construct a simple_component instance holding a new wrapped
        ///        instance
        template <typename... Ts>
        component(Ts&&... vs)
          : Component(std::forward<Ts>(vs)...)
        {}
    };
}}

#endif
