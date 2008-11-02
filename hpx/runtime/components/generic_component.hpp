//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_GENERIC_COMPONENT_OCT_12_2008_0947PM)
#define HPX_COMPONENTS_GENERIC_COMPONENT_OCT_12_2008_0947PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/stubs/generic_component.hpp>
#include <hpx/runtime/components/client_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ServerComponent>
    class generic_component 
      : public client_base<
            generic_component<ServerComponent>, 
            stubs::generic_component<ServerComponent>
        >
    {
    private:
        typedef client_base<
            generic_component<ServerComponent>, 
            stubs::generic_component<ServerComponent>
        > base_type;
        typedef typename base_type::result_type result_type;

    public:
        /// Create a client side representation for the existing 
        /// \a server#generic_component instance with the given global id 
        /// \a gid.
        generic_component(applier::applier& app, naming::id_type const& gid,
                bool freeonexit = false) 
          : base_type(app, gid, freeonexit)
        {
        }

        /// Invoke the action exposed by this generic component
        result_type eval(threads::thread_self& self)
        {
            return this->base_type::eval(self, gid_);
        }

        // bring in higher order eval functions
        #include <hpx/runtime/components/generic_component_eval.hpp>

    };

}}

#endif

