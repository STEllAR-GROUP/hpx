//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXAMPLES_BRIGHT_FUTURE_DATAFLOW_SERVER_DATAFLOW_TRIGGER_HPP
#define EXAMPLES_BRIGHT_FUTURE_DATAFLOW_SERVER_DATAFLOW_TRIGGER_HPP

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>

#include <examples/bright_future/dataflow/server/detail/dataflow_slot.hpp>
#include <examples/bright_future/dataflow/server/detail/component_wrapper.hpp>

namespace hpx { namespace lcos { namespace server
{
    /// The dataflow server side representation
    struct dataflow_trigger
        : components::managed_component_base<dataflow_trigger>
    {

        typedef dataflow_trigger wrapped_type;

        dataflow_trigger()
            : all_set(false)
            , slots_set(0)
            , slots_completed(~0u)
        {}

        ~dataflow_trigger()
        {
            trigger_targets();
            BOOST_FOREACH(detail::component_wrapper_base * c, triggers)
            {
                delete c;
            }
        }

        template <typename Result, typename RemoteResult>
        void add(dataflow_base<Result, RemoteResult> const & source)
        {
            typedef
                detail::component_wrapper<
                    detail::dataflow_slot<
                        dataflow_base<Result, RemoteResult>
                      , -1
                      , wrapped_type
                    >
                >
                component_type;
            LLCO_(info) <<
                "server::dataflow_trigger::add() {" << get_gid() << "}"
                ;
            component_type * w;
            {
                hpx::util::spinlock::scoped_lock l(mtx);
                w = new component_type(this, source, triggers.size());
                //slots_completed |= (1<<triggers.size());
                triggers.push_back(w);
            }
            (*w)->connect();
        }

        void set_trigger_size(unsigned size)
        {
            {
                hpx::util::spinlock::scoped_lock l(mtx);
                slots_completed = 0;
                for(unsigned i = 0; i < size; ++i)
                {
                    slots_completed |= (1<<i);
                }
            }
            trigger_targets();
        }

        typedef
            ::hpx::actions::action1<
                dataflow_trigger
              , 0
              , unsigned
              , &dataflow_trigger::set_trigger_size
            >
            set_trigger_size_action;

        void connect(naming::id_type const & target)
        {
            LLCO_(info) <<
                "server::dataflow_trigger::connect(" << target << ") {" << get_gid() << "}"
                ;
            hpx::util::spinlock::scoped_lock l(mtx);
            if(all_set)
            {
                typedef hpx::lcos::base_lco::set_event_action action_type;
                l.unlock();
                applier::apply<action_type>(target);
            }
            else
            {
                targets.push_back(target);
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
                hpx::util::spinlock::scoped_lock l(mtx);
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
                hpx::util::spinlock::scoped_lock l(mtx);
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
                applier::apply<action_type>(t[i]);
            }
        }

    private:
        bool all_set;
        boost::uint32_t slots_set;
        boost::uint32_t slots_completed;
        hpx::util::spinlock mtx;
        std::vector<detail::component_wrapper_base *> triggers;
        std::vector<naming::id_type> targets;
    };

    template <typename Result, typename RemoteResult>
    struct add_action
    {
        typedef
            hpx::actions::action1<
                dataflow_trigger
              , 0
              , dataflow_base<Result, RemoteResult> const &
              , &dataflow_trigger::add
            >
            type;
    };
}}}

#endif
