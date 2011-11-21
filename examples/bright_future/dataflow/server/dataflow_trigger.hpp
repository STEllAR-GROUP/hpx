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
            , slots_completed(0)
        {}

        ~dataflow_trigger()
        {
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

            component_type * w = new component_type(this, source, triggers.size());
            (*w)->connect();

            {
                hpx::util::spinlock::scoped_lock l(mtx);
                slots_completed |= (1<<triggers.size());
                triggers.push_back(w);
            }
        }

        void connect(naming::id_type const & target)
        {
            bool all_set_loc = false;
            {
                hpx::util::spinlock::scoped_lock l(mtx);
                all_set_loc = all_set;
            }
            if(all_set_loc)
            {
                typedef typename hpx::lcos::base_lco::set_event_action action_type;
                applier::apply<action_type>(target);
            }
            else
            {
                hpx::util::spinlock::scoped_lock l(mtx);
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
            bool all_set_loc = false;
            {
                typename hpx::util::spinlock::scoped_lock l(mtx);
                slots_set |= (1<<index);
                all_set_loc = (slots_set == slots_completed);
                all_set = all_set_loc;
            }
            if(all_set_loc)
            {
                trigger_targets();
            }
        }
        
        void trigger_targets()
        {
            std::vector<naming::id_type> t;
            {
                typename hpx::util::spinlock::scoped_lock l(mtx);
                if(all_set == false) return;
                std::swap(targets, t);
            }

            // Note: lco::set_result is a direct action, for this reason,
            //       the following loop will not be parallelized if the
            //       targets are local (which is ok)
            for (std::size_t i = 0; i < t.size(); ++i)
            {
                typedef typename hpx::lcos::base_lco::set_event_action action_type;
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
