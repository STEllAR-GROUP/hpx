//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_VALUE_OUT_ADAPTOR_OCT_17_2008_0956PM)
#define HPX_COMPONENTS_AMR_STENCIL_VALUE_OUT_ADAPTOR_OCT_17_2008_0956PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>

#include <boost/function.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    class stencil_value_out_adaptor
      : public components::detail::managed_component_base<
            stencil_value_out_adaptor
        >
    {
    private:
        typedef 
            boost::function<void(threads::thread_self&, naming::id_type*)>
        callback_function_type;

    public:
        stencil_value_out_adaptor(applier::applier& appl)
        {}

        /// set function to call whenever the result is requested
        void set_callback(callback_function_type eval)
        {
            eval_ = eval;
        }

        ///////////////////////////////////////////////////////////////////////
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            stencil_value_out_get_value = 0,
        };

        /// This is the main entry point of this component. Calling this 
        /// function (by applying the get_value) will return the value as 
        /// computed by the current time step.
        threads::thread_state 
        get_value (threads::thread_self& self, applier::applier& appl, 
            naming::id_type* result)
        {
            BOOST_ASSERT(eval_);      // must have been initialized
            eval_(self, result);
            return threads::terminated;
        }

        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action0<
            stencil_value_out_adaptor, naming::id_type, 
            stencil_value_out_get_value, &stencil_value_out_adaptor::get_value
        > get_value_action;

    private:
        callback_function_type eval_;
    };

}}}}

#endif

