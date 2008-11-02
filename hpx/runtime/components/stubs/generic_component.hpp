//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_GENERIC_COMPONENT_OCT_12_2008_0937PM)
#define HPX_COMPONENTS_STUBS_GENERIC_COMPONENT_OCT_12_2008_0937PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/generic_component.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace stubs
{
    namespace detail
    {
        template <typename Action, typename Result>
        struct eval
        {
            static Result call(threads::thread_self& self, 
                applier::applier& appl, naming::id_type const& gid)
            {
                lcos::eager_future<Action, Result> f(appl, gid);
                return f.get(self);
            }

            template <typename ParameterBlock>
            static Result call(threads::thread_self& self, 
                applier::applier& appl, naming::id_type const& gid, 
                ParameterBlock const& params)
            {
                lcos::eager_future<Action, Result> f(appl, gid, params);
                return f.get(self);
            }
        };

        template <typename Action>
        struct eval<Action, void>
        {
            static void call(threads::thread_self&, 
                applier::applier& appl, naming::id_type const& gid)
            {
                appl.apply<Action>(gid);
            }

            template <typename ParameterBlock>
            static void call(threads::thread_self&, applier::applier& appl, 
                naming::id_type const& gid, ParameterBlock const& params)
            {
                appl.apply<Action>(gid, params);
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename ServerComponent, 
        typename Action = typename ServerComponent::eval_action,
        typename Result = typename ServerComponent::result_type, 
        typename Parameters = typename ServerComponent::parameter_block_type
    > class generic_component;

    ///////////////////////////////////////////////////////////////////////////
    template <typename ServerComponent, typename Action, typename Result, 
        typename Parameters>
    class generic_component : public stub_base<ServerComponent>
    {
    protected:
        typedef Action action_type;
        typedef Result result_type;
        typedef Parameters params_type;

    public:
        /// Create a client side representation for any existing 
        /// \a server#generic_component0 instance
        generic_component(applier::applier& app) 
          : stub_base<ServerComponent>(app)
        {}

        /// Invoke the action exposed by this generic component
        static result_type 
        eval(threads::thread_self& self, applier::applier& appl, 
            naming::id_type const& targetgid)
        {
            typedef typename ServerComponent::eval_action action_type;
            return detail::eval<action_type, result_type>::call(self, appl, 
                targetgid);
        }

        result_type eval(threads::thread_self& self, 
            naming::id_type const& targetgid)
        {
            return eval(self, this->appl_, targetgid);
        }

        // bring in higher order eval functions
        #include <hpx/runtime/components/stubs/generic_component_eval.hpp>

    };

}}}

#endif

