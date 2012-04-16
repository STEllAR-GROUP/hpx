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
#include <hpx/components/dataflow/server/detail/dataflow_trigger_slot.hpp>

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
        typedef base_type::wrapping_type wrapping_type;

        static components::component_type get_component_type()
        {
            return components::get_component_type<dataflow_trigger>();
        }
        static void set_component_type(components::component_type type)
        {
            components::set_component_type<dataflow_trigger>(type);
        }

        dataflow_trigger()
            : slots_complete(0)
        {
            BOOST_ASSERT(false);
        }

        dataflow_trigger(component_type * back_ptr)
            : base_type(back_ptr)
            , slots_complete(0)
        {
            BOOST_ASSERT(false);
        }

        static boost::uint64_t calc_complete_mask(std::size_t N)
        {
            boost::uint64_t r = 0;
            for(std::size_t i = 0; i < N; ++i)
            {
                r |= (std::size_t(1)<<i);
            }
            return r;
        }

        dataflow_trigger(
            component_type * back_ptr
          , std::vector<hpx::lcos::dataflow_base<void> > const & trigger
        )
          : base_type(back_ptr)
          , slots_set(0)
          , slots_complete(calc_complete_mask(trigger.size()))
        {
            typedef
                hpx::lcos::dataflow_base<void>
                dataflow_type;

            typedef
                detail::dataflow_trigger_slot<dataflow_type, dataflow_trigger>
                dataflow_slot_type;

            typedef
                detail::component_wrapper<dataflow_slot_type>
                component_type;

            std::size_t slot_idx = 0;
            future_slots.reserve(trigger.size());

            BOOST_FOREACH(dataflow_type const &d, trigger)
            {
                component_type * c
                    = new component_type(this, d, slot_idx);
                (*c)->connect();
                future_slots.push_back(c);
                ++slot_idx;
            }
        }

        void finalize()
        {}

        ~dataflow_trigger()
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
                hpx::apply<action_type>(target);
            }
            BOOST_ASSERT(targets.empty());
        }

        template <typename T>
        void set_slot(T, std::size_t slot)
        {
            bool trigger = false;
            std::vector<naming::id_type> tmp;
            {
                lcos::local::spinlock::scoped_lock l(mtx);
                if(slots_set != slots_complete)
                {
                    slots_set |= (std::size_t(1)<<slot);
                    trigger = (slots_set == slots_complete);
                    if(trigger) std::swap(tmp, targets);
                }
            }

            if(trigger)
            {
                data.set(data_type(util::unused_type()));
                BOOST_FOREACH(naming::id_type const & target, tmp)
                {
                    typedef hpx::lcos::base_lco::set_event_action action_type;
                    BOOST_ASSERT(target);
                    hpx::apply<action_type>(target);
                }
                BOOST_ASSERT(targets.empty());
            }
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
                    hpx::apply<action_type>(target);
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

        std::vector<detail::component_wrapper_base *> future_slots;
        boost::uint64_t slots_set;
        const boost::uint64_t slots_complete;
    };
}}}

#endif
