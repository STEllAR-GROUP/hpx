//  Copyright (c) 2011 Matt Anderson
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_7D3E9527_A993_42F2_B8AC_670F0955A64B)
#define HPX_7D3E9527_A993_42F2_B8AC_670F0955A64B

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/lcos/local_mutex.hpp>

namespace random_mem_access { namespace server
{
    class HPX_COMPONENT_EXPORT element
      : public hpx::components::managed_component_base<element>
    {
    public:
        typedef hpx::lcos::local_mutex mutex_type;

        enum actions
        {
            element_init = 0,
            element_add = 1,
            element_print = 2
        };

        element()
          : arg_(0), arg_init_(0)
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Initialize the element. 
        void init(std::size_t i);

        /// Increment the element's value.
        void add();

        /// Print the current value of the element.
        void print();

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action1<
            // Component server type.
            element,
            // Action code. 
            element_init,
            // Arguments of this action.
            std::size_t,
            // Method bound to this action.
            &element::init
        > init_action;

        typedef hpx::actions::action0<
            // Component server type.
            element,
            // Action code. 
            element_add,
            // Method bound to this action.
            &element::add
        > add_action;

        typedef hpx::actions::action0<
            // Component server type.
            element,
            // Action code. 
            element_print,
            // Method bound to this action.
            &element::print
        > print_action;

    private:
        std::size_t arg_;
        std::size_t arg_init_;
        mutex_type mtx_;
    };
}}

#endif

