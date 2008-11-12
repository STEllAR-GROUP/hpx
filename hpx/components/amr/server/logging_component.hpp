//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_SERVER_LOGGING_COMPONENT_NOV_10_2008_0651PM)
#define HPX_COMPONENTS_AMR_SERVER_LOGGING_COMPONENT_NOV_10_2008_0651PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT logging_component
      : public simple_component_base<logging_component>
    {
    private:
        typedef simple_component_base<logging_component> base_type;

    public:
        logging_component(threads::thread_self& self, applier::applier& appl)
          : base_type(self, appl)
        {}

        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this 
        // component
        typedef amr::server::logging_component wrapping_type;

        enum actions
        {
            logging_component_logentry = 0,
        };

        ///////////////////////////////////////////////////////////////////////
        virtual threads::thread_state logentry(threads::thread_self&, 
            applier::applier&, naming::id_type const&)
        {
            // This shouldn't ever be called. If you're seeing this assertion 
            // you probably forgot to overload this function in your stencil 
            // class.
            BOOST_ASSERT(false);
            return threads::terminated;
        }

        threads::thread_state logentry_nonvirt(threads::thread_self& self, 
            applier::applier& appl, naming::id_type const& memblock)
        {
            return logentry(self, appl, memblock);
        }

        /// Each of the exposed functions needs to be encapsulated into an action
        /// type, allowing to generate all required boilerplate code for threads,
        /// serialization, etc.
        ///
        /// The \a set_result_action may be used to trigger any LCO instances
        /// while carrying an additional parameter of any type.
        ///
        /// \param Result [in] The type of the result to be transferred back to 
        ///               this LCO instance.
        typedef hpx::actions::action1<
            logging_component, logging_component_logentry, 
            naming::id_type const&, &logging_component::logentry_nonvirt
        > logentry_action;
    };

}}}}

#endif
