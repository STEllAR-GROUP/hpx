//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2015-2017 Thomas Heller
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_COMPONENTS_SERVER_COMPONENT_HPP
#define HPX_RUNTIME_COMPONENTS_SERVER_COMPONENT_HPP

#include <hpx/config.hpp>
#include <hpx/util/assert.hpp>

#include <cstddef>
#include <new>
#include <utility>

namespace hpx { namespace components {

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Component>
        struct simple_heap_factory
        {
            static void* alloc(std::size_t count)
            {
                HPX_ASSERT(1 == count);
                return ::operator new(sizeof(Component));
            }
            static void free(void* p, std::size_t count)
            {
                HPX_ASSERT(1 == count);
                ::operator delete(p);
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    class component : public Component
    {
    public:
        typedef Component type_holder;
        typedef component<Component> component_type;
        typedef component_type derived_type;
        typedef detail::simple_heap_factory<component_type> heap_type;

        /// \brief Construct a simple_component instance holding a new wrapped
        ///        instance
        template <typename... Ts>
        component(Ts&&... vs)
          : Component(std::forward<Ts>(vs)...)
        {}

        /// \brief  The function \a create is used for allocation and
        ///         initialization of instances of the derived components.
        static component_type* create(std::size_t count)
        {
            // simple components can be created individually only
            HPX_ASSERT(1 == count);
            return new component_type();    //-V572
        }

        /// \brief  The function \a destroy is used for destruction and
        ///         de-allocation of instances of the derived components.
        static void destroy(Component* p, std::size_t count = 1)
        {
            // simple components can be deleted individually only
            HPX_ASSERT(1 == count);
            p->finalize();
            delete p;
        }
    };
}}

#endif
