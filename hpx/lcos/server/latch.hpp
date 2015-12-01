//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_SERVER_LATCH_APR_19_2015_0956AM)
#define HPX_LCOS_SERVER_LATCH_APR_19_2015_0956AM

#include <hpx/hpx_fwd.hpp>

#include <hpx/lcos/local/latch.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace server
{
    /// A latch can be used to synchronize a specific number of threads,
    /// blocking all of the entering threads until all of the threads have
    /// entered the latch.
    class latch
      : public lcos::base_lco_with_value<bool, std::ptrdiff_t>,
        public components::managed_component_base<latch>
    {
    private:
        typedef components::managed_component_base<latch> base_type;

    public:
        typedef lcos::base_lco_with_value<bool, std::ptrdiff_t> base_type_holder;

        // disambiguate base classes
        using base_type::finalize;
        using base_type::decorate_action;
        using base_type::schedule_thread;
        using base_type::is_target_valid;

        typedef base_type::wrapping_type wrapping_type;

        static components::component_type get_component_type()
        {
            return components::component_latch;
        }
        static void set_component_type(components::component_type) {}

    public:
        // This is the component type id. Every component type needs to have an
        // embedded enumerator 'value' which is used by the generic action
        // implementation to associate this component with a given action.
        enum { value = components::component_latch };

        latch()
          : latch_(0)
        {}

        latch(std::ptrdiff_t number_of_threads)
          : latch_(number_of_threads)
        {}

        // standard LCO action implementations

        /// The function \a set_event will block the number of entering
        /// \a threads (as given by the constructor parameter \a number_of_threads),
        /// releasing all waiting threads as soon as the last \a thread
        /// entered this function.
        ///
        /// This is invoked whenever the count_down_and_wait() function is called
        void set_event()
        {
            latch_.count_down_and_wait();
        }

        /// This is invoked whenever the count_down() function is called
        void set_value(std::ptrdiff_t && n) //-V669
        {
            latch_.count_down(n);
        }

        /// This is invoked whenever the is_ready() function is called
        bool get_value(hpx::error_code &)
        {
            return latch_.is_ready();
        }

        /// The \a function set_exception is called whenever a
        /// \a set_exception_action is applied on an instance of a LCO. This
        /// function just forwards to the virtual function \a set_exception,
        /// which is overloaded by the derived concrete LCO.
        ///
        /// \param e      [in] The exception encapsulating the error to report
        ///               to this LCO instance.
        void set_exception(boost::exception_ptr const& e)
        {
            try {
                latch_.abort_all();
                boost::rethrow_exception(e);
            }
            catch (boost::exception const& be) {
                // rethrow again, but this time using the native hpx mechanics
                HPX_THROW_EXCEPTION(hpx::no_success, "latch::set_exception",
                    boost::diagnostic_information(be));
            }
        }

        typedef hpx::components::server::create_component_action<
                latch, std::ptrdiff_t
            > create_component_action;

        // additional functionality
        void wait() const
        {
            latch_.wait();
        }
        HPX_DEFINE_COMPONENT_ACTION(latch, wait, wait_action);

    private:
        lcos::local::latch latch_;
    };
}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::server::latch::create_component_action,
    hpx_lcos_server_latch_create_component_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::server::latch::wait_action,
    hpx_lcos_server_latch_wait_action)

#endif

