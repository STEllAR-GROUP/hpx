//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matthew Anderson
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
#include <hpx/config/function.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT stencil_value_out_adaptor
      : public components::detail::managed_component_base<
            stencil_value_out_adaptor
        >
    {
    private:
        typedef HPX_STD_FUNCTION<naming::id_type()> callback_function_type;
        typedef components::detail::managed_component_base<
            stencil_value_out_adaptor
        > base_type;

    public:
        stencil_value_out_adaptor(callback_function_type eval = callback_function_type())
          : eval_(eval)
        {
            if (component_invalid == base_type::get_component_type()) {
                // first call to get_component_type, ask AGAS for a unique id
                base_type::set_component_type(applier::get_applier().get_agas_client().
                    get_component_id("dataflow_stencil_value_out_adaptor"));
            }
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
        naming::id_type get_value ()
        {
            BOOST_ASSERT(eval_);      // must have been initialized
            return eval_();
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

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::amr::server::stencil_value_out_adaptor::get_value_action,
    dataflow_stencil_value_out_get_value_action);

#endif

