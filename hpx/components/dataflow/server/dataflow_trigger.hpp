//  Copyright (c) 2011-2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_SERVER_DATAFLOW_TRIGGER_HPP
#define HPX_LCOS_DATAFLOW_SERVER_DATAFLOW_TRIGGER_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/components/dataflow/dataflow_base.hpp>

namespace hpx { namespace lcos { namespace server
{
    /// The dataflow server side representation
    struct HPX_COMPONENT_EXPORT dataflow_trigger
        : base_lco
        , components::managed_component_base<
            dataflow_trigger
          , hpx::components::detail::this_type
          , hpx::traits::construct_with_back_ptr
        >
    {
        typedef
            components::managed_component_base<
                dataflow_trigger
              , hpx::components::detail::this_type
              , hpx::traits::construct_with_back_ptr
            >
            base_type;
        typedef hpx::components::managed_component<dataflow_trigger> component_type;
        
        typedef util::value_or_error<util::unused_type> data_type;

        // disambiguate base classes
        typedef base_lco base_type_holder;
        using base_type::finalize;
        typedef typename base_type::wrapping_type wrapping_type;

        static components::component_type get_component_type()
        {
            return components::get_component_type<dataflow_trigger>();
        }
        static void set_component_type(components::component_type type)
        {
            components::set_component_type<dataflow_trigger>(type);
        }

        void wait_for_trigger(std::vector<hpx::lcos::dataflow_base<void> > const & trigger)
        {
            // We first convert every dataflow_base element into a future
            // which means connecting to the triggers
            std::vector<lcos::future<void> > trigger_future;
            trigger_future.reserve(trigger.size());
            
            LLCO_(info)
                << "hpx::lcos::server::dataflow_trigger: converting dataflow to futures start\n";

            //BOOST_FOREACH(hpx::lcos::dataflow_base<void> const & t, trigger)
            for(std::size_t i; i < trigger.size(); ++i)
            {
                trigger_future.push_back(trigger[i].get_future());
                new hpx::lcos::dataflow_base<void>(trigger[i]);
            }
            
            LLCO_(info)
                << "hpx::lcos::server::dataflow_trigger: converting dataflow to futures finished\n";

            // After connecting, we can wait for the trigger
            // This can be in serial, because we have to wait for all trigger
            // to fire anyway, by getting the futures first, we already
            // connected to the dataflows asynchronously.
            //BOOST_FOREACH(lcos::future<void> const & f, trigger_future)
            for(std::size_t i; i < trigger_future.size(); ++i)
            {
            
                LLCO_(info)
                    << "hpx::lcos::server::dataflow_trigger: waiting for trigger " << i << "\n";
                // TODO: error handling
                trigger_future[i].get();
                LLCO_(info)
                    << "hpx::lcos::server::dataflow_trigger: got trigger " << i << "\n";
            }

            // after all triggers completed the operation, we fire all already
            // connected dataflow targets
            std::vector<naming::id_type> tmp;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(tmp, targets);
            }

            // we set the future_data, so that targets connecting
            // in the meantime can be notified as well
            data.set(data_type(util::unused_type()));

            BOOST_FOREACH(naming::id_type const & target, tmp)
            {
                typedef hpx::lcos::base_lco::set_event_action action_type;
                BOOST_ASSERT(target);
                applier::apply<action_type>(target);
            }
        }

        dataflow_trigger(component_type * back_ptr)
            : base_type(back_ptr)
        {
            BOOST_ASSERT(false);
        }

        dataflow_trigger(component_type * back_ptr, std::vector<hpx::lcos::dataflow_base<void> > const & trigger)
            : base_type(back_ptr)
        {
            applier::register_thread_nullary(
                HPX_STD_BIND(&dataflow_trigger::wait_for_trigger, this, trigger),
                "dataflow_trigger::wait_for_trigger");
        }

        void finalize()
        {
            LLCO_(info)
                << "hpx::lcos::server::dataflow_trigger: finalize\n";
            data_type d;
            data.read(d);
            std::vector<naming::id_type> tmp;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                std::swap(tmp, targets);
            }

            BOOST_FOREACH(naming::id_type const & target, tmp)
            {
                typedef hpx::lcos::base_lco::set_event_action action_type;
                BOOST_ASSERT(target);
                applier::apply<action_type>(target);
            }
        }

        ~dataflow_trigger()
        {
            BOOST_ASSERT(targets.empty());
        }

        void connect(naming::id_type const & target)
        {
            LLCO_(info) <<
                "server::dataflow_trigger::connect(" << target << ") {" << get_gid() << "}"
                ;
            {
                if(!data.is_empty())
                {
                    typedef hpx::lcos::base_lco::set_event_action action_type;
                    BOOST_ASSERT(target);
                    applier::apply<action_type>(target);
                }
                else
                {
                    lcos::local::spinlock::scoped_lock l(mtx);
                    targets.push_back(target);
                }
            }
        }

        void set_event() {}

    private:
        util::full_empty<data_type> data;
        // TODO: investigate if lockfree fifo would be better that std::vector + spinlock
        lcos::local::spinlock mtx;
        std::vector<naming::id_type> targets;
    };
}}}

#endif
