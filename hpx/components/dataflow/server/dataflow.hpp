//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_SERVER_DATAFLOW_HPP
#define HPX_LCOS_DATAFLOW_SERVER_DATAFLOW_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/components/dataflow/server/detail/dataflow_impl.hpp>
#include <hpx/components/dataflow/server/detail/component_wrapper.hpp>
#include <hpx/util/decay.hpp>

namespace hpx { namespace lcos { namespace server
{
    /// The dataflow server side representation
    struct HPX_COMPONENT_EXPORT dataflow
        : base_lco
        , components::managed_component_base<
            dataflow
          , hpx::components::detail::this_type
          , hpx::traits::construct_with_back_ptr
        >
    {
        typedef
            components::managed_component_base<
                dataflow
              , hpx::components::detail::this_type
              , hpx::traits::construct_with_back_ptr
            >
            base_type;
        typedef hpx::components::managed_component<dataflow> component_type;

        // disambiguate base classes
        typedef base_lco base_type_holder;
        using base_type::finalize;
        typedef base_type::wrapping_type wrapping_type;

        static components::component_type get_component_type()
        {
            return components::get_component_type<dataflow>();
        }
        static void set_component_type(components::component_type type)
        {
            components::set_component_type<dataflow>(type);
        }

        void finalize()
        {
        }

        ~dataflow()
        {
            LLCO_(info) << "~server::dataflow::dataflow()";

            HPX_ASSERT(component_ptr);
            if (component_ptr) {
                component_ptr->finalize();
                delete component_ptr;
            }
            detail::update_destructed_count();
        }

        /// init initializes the dataflow, it creates a dataflow_impl object
        /// that holds old type information and does the remaining processing
        /// of managing the dataflow.
        /// init is a variadic function. The first template parameter denotes
        /// the Action that needs to get spawned once all arguments are
        /// computed
        template <typename Action, typename ...Ts>
        void init(
            naming::id_type const & target
          , Ts&&... vs
        )
        {
            typedef
                detail::dataflow_impl<
                    Action
                  , typename traits::promise_local_result<
                        typename Action::result_type>::type(
                            typename util::decay<Ts>::type...)
                >
                wrapped_type;

            typedef
                detail::component_wrapper<wrapped_type>
                component_type;

            LLCO_(info)
                << "server::dataflow::init() " << get_gid();

            component_type * w = new component_type(target, mtx, targets);
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                component_ptr = w;
            }
            (*w)->init(std::forward<Ts>(vs)...);

            detail::update_initialized_count();
        }

        dataflow()
        {
            HPX_ASSERT(false);
        }

        dataflow(component_type * back_ptr)
            : base_type(back_ptr)
        {
            HPX_ASSERT(false);
        }

        template <typename Action, typename ...Ts>
        dataflow(
            component_type * back_ptr
          , Action
          , naming::id_type const & target
          , Ts&&... vs
        )
            : base_type(back_ptr)
            , component_ptr(0)
        {
            /*
            applier::register_thread(
                util::bind(&dataflow::init<
                        typename Action::type, Ts...
                    >
                  , this
                  , target
                  , std::forward<Ts>(vs)...
                )
              , "hpx::lcos::server::dataflow::init<>"
            );
            */
            init<typename Action::type>(target, std::forward<Ts>(vs)...);
        }

        /// the connect function is used to connect the current dataflow
        /// to the specified target lco
        void connect(naming::id_type const & target)
        {
            LLCO_(info) <<
                "hpx::lcos::server::dataflow::connect(" << target << ") {" << get_gid() << "}"
                ;
            {
                lcos::local::spinlock::scoped_lock l(mtx);

                // wait until component_ptr is initialized.
                if(component_ptr == 0)
                {
                    targets.push_back(target);
                    return;
                }
            }
            (*component_ptr)->connect_nonvirt(target);
        }

        void set_event() {}

    private:
        detail::component_wrapper_base * component_ptr;
        lcos::local::spinlock mtx;
        std::vector<naming::id_type> targets;
    };
}}}

#endif
