//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_SERVER_DATAFLOW_TRIGGER_HPP
#define HPX_LCOS_DATAFLOW_SERVER_DATAFLOW_TRIGGER_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/components/dataflow/dataflow_base.hpp>
#include <hpx/components/dataflow/server/detail/dataflow_slot.hpp>
#include <hpx/components/dataflow/server/detail/component_wrapper.hpp>

namespace hpx { namespace lcos { namespace server
{
    /// The dataflow server side representation
    struct HPX_COMPONENT_EXPORT dataflow_trigger
        : components::managed_component_base<
            dataflow_trigger
          , hpx::components::detail::this_type
          , hpx::components::detail::construct_with_back_ptr
        >
    {
        typedef dataflow_trigger wrapped_type;
        typedef
            components::managed_component_base<
                dataflow_trigger
              , hpx::components::detail::this_type
              , hpx::components::detail::construct_with_back_ptr
            >
            base_type;

        typedef hpx::components::managed_component<dataflow_trigger> component_type;

        void init(std::vector<dataflow_base<void> > const & trigger)
        {
            typedef
                detail::component_wrapper<
                    detail::dataflow_slot<
                        dataflow_base<void>
                      , -1
                      , wrapped_type
                    >
                >
                component_type;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                slots_completed = 0;
                for(unsigned i = 0; i < trigger.size(); ++i)
                {
                    slots_completed |= (1<<i);
                }
            }
            triggers.reserve(trigger.size());
            for(unsigned i = 0; i < trigger.size(); ++i)
            {
                component_type * w = new component_type(this, trigger.at(i), i);
                (*w)->connect();
                triggers.push_back(w);
            }
        }

        dataflow_trigger(component_type * back_ptr)
            : base_type(back_ptr)
            , all_set(false)
            , slots_set(0)
            , slots_completed(~0u)
        {
            BOOST_ASSERT(false);
        }

        dataflow_trigger(component_type * back_ptr, std::vector<dataflow_base<void> > const & trigger)
            : base_type(back_ptr)
            , all_set(false)
            , slots_set(0)
            , slots_completed(~0u)
        {
            applier::register_thread(
                HPX_STD_BIND(&dataflow_trigger::init, this, trigger),
                "dataflow_trigger::init<>");
        }

        ~dataflow_trigger()
        {
            trigger_targets();
            BOOST_ASSERT(all_set);
            BOOST_FOREACH(detail::component_wrapper_base * c, triggers)
            {
                delete c;
            }
        }

        void connect(naming::id_type const & target)
        {
            LLCO_(info) <<
                "server::dataflow_trigger::connect(" << target << ") {" << get_gid() << "}"
                ;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(all_set)
                {
                    typedef hpx::lcos::base_lco::set_event_action action_type;
                    l.unlock();
                    BOOST_ASSERT(target);
                    applier::apply<action_type>(target);
                }
                else
                {
                    targets.push_back(target);
                }
            }
        }

        typedef
            ::hpx::actions::action1<
                dataflow_trigger
              , 0
              , naming::id_type const &
              , &dataflow_trigger::connect
            >
            connect_action;

        void set_slot(unsigned index)
        {
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_completed)
                {
                    slots_set |= (1<<index);
                }
            }
            trigger_targets();
        }

        void trigger_targets()
        {
            std::vector<naming::id_type> t;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                all_set = (slots_set == slots_completed);
                if(all_set == false) return;
                std::swap(targets, t);
            }

            // Note: lco::set_result is a direct action, for this reason,
            //       the following loop will not be parallelized if the
            //       targets are local (which is ok)
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef hpx::lcos::base_lco::set_event_action action_type;
                BOOST_ASSERT(t[i]);
                applier::apply<action_type>(t[i]);
            }
        }

    private:
        bool all_set;
        boost::uint32_t slots_set;
        boost::uint32_t slots_completed;
        lcos::local::spinlock mtx;
        std::vector<detail::component_wrapper_base *> triggers;
        std::vector<naming::id_type> targets;
    };
}}}

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::lcos::server::dataflow_trigger::connect_action
  , dataflow_trigger_type_connect_action)

#endif
