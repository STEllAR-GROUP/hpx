//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_SERVER_BARRIER_MAR_10_2010_0310PM)
#define HPX_LCOS_SERVER_BARRIER_MAR_10_2010_0310PM

#include <hpx/hpx_fwd.hpp>

#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/local/detail/condition_variable.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>

#include <boost/thread/locks.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace server
{
    /// A barrier can be used to synchronize a specific number of threads,
    /// blocking all of the entering threads until all of the threads have
    /// entered the barrier.
    class barrier
      : public lcos::base_lco
      , public components::managed_component_base<barrier>
    {
    public:
        typedef lcos::base_lco base_type_holder;

    private:
        typedef components::managed_component_base<barrier> base_type;

        typedef hpx::lcos::local::spinlock mutex_type;
        mutex_type mtx_;

    public:
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = components::component_barrier };

        barrier()
          : number_of_threads_(1)
        {}

        barrier(std::size_t number_of_threads)
          : number_of_threads_(number_of_threads)
        {}

        // disambiguate base classes
        using base_type::finalize;
        typedef base_type::wrapping_type wrapping_type;

        static components::component_type get_component_type()
        {
            return components::component_barrier;
        }
        static void set_component_type(components::component_type) {}

        // standard LCO action implementations

        /// The function \a set_event will block the number of entering
        /// \a threads (as given by the constructor parameter \a number_of_threads),
        /// releasing all waiting threads as soon as the last \a thread
        /// entered this function.
        void set_event()
        {
            boost::unique_lock<mutex_type> l(mtx_);
            if (cond_.size(l) < number_of_threads_-1) {
                cond_.wait(l, "barrier::set_event");
            }
            else {
                cond_.notify_all(std::move(l));
            }
        }

        /// The \a function set_exception is called whenever a
        /// \a set_exception_action is applied on an instance of a LCO. This
        /// function just forwards to the virtual function \a set_exception, which
        /// is overloaded by the derived concrete LCO.
        ///
        /// \param e      [in] The exception encapsulating the error to report
        ///               to this LCO instance.
        void set_exception(boost::exception_ptr const& e)
        {
            try {
                boost::unique_lock<mutex_type> l(mtx_);
                cond_.abort_all(std::move(l));

                boost::rethrow_exception(e);
            }
            catch (boost::exception const& be) {
                // rethrow again, but this time using the native hpx mechanics
                HPX_THROW_EXCEPTION(hpx::no_success, "barrier::set_exception",
                    boost::diagnostic_information(be));
            }
        }

        typedef
            hpx::components::server::create_component_action<
                barrier
              , std::size_t
            >
            create_component_action;

    private:
        std::size_t const number_of_threads_;
        local::detail::condition_variable cond_;
    };
}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::server::barrier::create_component_action
  , hpx_lcos_server_barrier_create_component_action
)

#endif

