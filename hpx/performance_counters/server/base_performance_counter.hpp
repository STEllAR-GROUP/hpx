//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_SERVER_BASE_MAR_03_2009_0741M)
#define HPX_PERFORMANCE_COUNTERS_SERVER_BASE_MAR_03_2009_0741M

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/performance_counter.hpp>

#include <boost/detail/atomic_count.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    class base_performance_counter
      : public hpx::performance_counters::performance_counter
    {
    protected:
        /// the following functions are not implemented by default, they will
        /// just throw
        virtual void reset_counter_value()
        {
            HPX_THROW_EXCEPTION(invalid_status, "reset_counter_value",
                "reset_counter_value is not implemented for this counter");
        }

        virtual void set_counter_value(counter_value const& /*value*/)
        {
            HPX_THROW_EXCEPTION(invalid_status, "set_counter_value",
                "set_counter_value is not implemented for this counter");
        }

        virtual bool start()
        {
            return false;
        }

        virtual bool stop()
        {
            return false;
        }

        virtual counter_info get_counter_info() const
        {
            return info_;
        }

    public:
        base_performance_counter() : invocation_count_(0) {}
        base_performance_counter(counter_info const& info)
          : info_(info), invocation_count_(0)
        {}

        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this
        // component
        typedef components::managed_component<base_performance_counter> wrapping_type;
        typedef base_performance_counter base_type_holder;

        /// \brief finalize() will be called just before the instance gets
        ///        destructed
        void finalize() {}

        static components::component_type get_component_type()
        {
            return components::get_component_type<wrapping_type>();
        }
        static void set_component_type(components::component_type t)
        {
            components::set_component_type<wrapping_type>(t);
        }

        ///////////////////////////////////////////////////////////////////////
        counter_info get_counter_info_nonvirt() const
        {
            return this->get_counter_info();
        }

        counter_value get_counter_value_nonvirt()
        {
            return this->get_counter_value();
        }

        void set_counter_value_nonvirt(counter_value const& info)
        {
            this->set_counter_value(info);
        }

        void reset_counter_value_nonvirt()
        {
            this->reset_counter_value();
        }

        bool start_nonvirt()
        {
            return this->start();
        }

        bool stop_nonvirt()
        {
            return this->stop();
        }

        /// Each of the exposed functions needs to be encapsulated into an action
        /// type, allowing to generate all required boilerplate code for threads,
        /// serialization, etc.

        /// The \a get_counter_info_action retrieves a performance counters
        /// information.
        typedef hpx::actions::result_action0<
            base_performance_counter const, counter_info,
            &base_performance_counter::get_counter_info_nonvirt
        > get_counter_info_action;

        /// The \a get_counter_value_action queries the value of a performance
        /// counter.
        typedef hpx::actions::result_action0<
            base_performance_counter, counter_value,
            &base_performance_counter::get_counter_value_nonvirt
        > get_counter_value_action;

        /// The \a set_counter_value_action
        typedef hpx::actions::action1<
            base_performance_counter,
            counter_value const&,
            &base_performance_counter::set_counter_value_nonvirt
        > set_counter_value_action;

        /// The \a reset_counter_value_action
        typedef hpx::actions::action0<
            base_performance_counter,
            &base_performance_counter::reset_counter_value_nonvirt
        > reset_counter_value_action;

        /// The \a start_action
        typedef hpx::actions::result_action0<
            base_performance_counter, bool,
            &base_performance_counter::start_nonvirt
        > start_action;

        /// The \a stop_action
        typedef hpx::actions::result_action0<
            base_performance_counter, bool,
            &base_performance_counter::stop_nonvirt
        > stop_action;

        /// This is the default hook implementation for decorate_action which
        /// does no hooking at all.
        static HPX_STD_FUNCTION<threads::thread_function_type>
        wrap_action(HPX_STD_FUNCTION<threads::thread_function_type> f,
            naming::address::address_type)
        {
            return boost::move(f);
        }

    protected:
        hpx::performance_counters::counter_info info_;
        boost::detail::atomic_count invocation_count_;
    };
}}}

HPX_ACTION_HAS_CRITICAL_PRIORITY(
    hpx::performance_counters::server::base_performance_counter::get_counter_info_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::performance_counters::server::base_performance_counter::get_counter_info_action,
    performance_counter_get_counter_info_action)

HPX_ACTION_HAS_CRITICAL_PRIORITY(
    hpx::performance_counters::server::base_performance_counter::get_counter_value_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::performance_counters::server::base_performance_counter::get_counter_value_action,
    performance_counter_get_counter_value_action)

HPX_ACTION_HAS_CRITICAL_PRIORITY(
    hpx::performance_counters::server::base_performance_counter::set_counter_value_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::performance_counters::server::base_performance_counter::set_counter_value_action,
    performance_counter_set_counter_value_action)

HPX_ACTION_HAS_CRITICAL_PRIORITY(
    hpx::performance_counters::server::base_performance_counter::reset_counter_value_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::performance_counters::server::base_performance_counter::reset_counter_value_action,
    performance_counter_reset_counter_value_action)

HPX_ACTION_HAS_CRITICAL_PRIORITY(
    hpx::performance_counters::server::base_performance_counter::start_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::performance_counters::server::base_performance_counter::start_action,
    performance_counter_start_action)

HPX_ACTION_HAS_CRITICAL_PRIORITY(
    hpx::performance_counters::server::base_performance_counter::stop_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::performance_counters::server::base_performance_counter::stop_action,
    performance_counter_stop_action)

HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    hpx::performance_counters::counter_info,
    counter_info)
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    hpx::performance_counters::counter_value,
    counter_value)

#endif

