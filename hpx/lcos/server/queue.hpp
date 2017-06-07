//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_SERVER_QUEUE_FEB_09_2011_1204PM)
#define HPX_LCOS_SERVER_QUEUE_FEB_09_2011_1204PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_QUEUE_COMPATIBILITY)
#include <hpx/error_code.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/lcos/local/detail/condition_variable.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/component_base.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/traits/get_remote_result.hpp>
#include <hpx/util/detail/count_num_args.hpp>

#include <boost/exception_ptr.hpp>

#include <memory>
#include <mutex>
#include <queue>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace server
{
    /// A queue can be used to 'collect' (queue) a number of incoming values
    template <typename ValueType, typename RemoteType = ValueType>
    class queue;

    ///////////////////////////////////////////////////////////////////////////
    template <typename ValueType, typename RemoteType>
    class queue
      : public lcos::base_lco_with_value<ValueType, RemoteType>
      , public components::managed_component_base<queue<ValueType, RemoteType> >
    {
    public:
        typedef lcos::base_lco_with_value<ValueType, RemoteType> base_type_holder;

    private:
        typedef lcos::local::spinlock mutex_type;
        typedef components::managed_component_base<queue> base_type;

        typedef std::queue<ValueType> queue_type;

    public:
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        queue()
        {}

        ~queue()
        {
            HPX_ASSERT(queue_.empty());
        }

        // disambiguate base classes
        using base_type::finalize;
        typedef typename base_type::wrapping_type wrapping_type;

        static components::component_type get_component_type()
        {
            return components::get_component_type<queue>();
        }
        static void set_component_type(components::component_type type)
        {
            components::set_component_type<queue>(type);
        }

        // standard LCO action implementations

        /// Add a value to the queue.
        void set_value (RemoteType && result)
        {
            // push back the new value onto the queue
            std::unique_lock<mutex_type> l(mtx_);
            queue_.push(
                traits::get_remote_result<ValueType, RemoteType>::call(
                    std::move(result)));

            // resume the first thread waiting to pick up that value
            cond_.notify_one(std::move(l));
        }

        /// The \a function set_exception is called whenever a
        /// \a set_exception_action is applied on an instance of a LCO.
        ///
        /// \param e      [in] The exception encapsulating the error to report
        ///               to this LCO instance.
        void set_exception(boost::exception_ptr const& /*e*/)
        {
            std::unique_lock<mutex_type> l(mtx_);
            cond_.abort_all(std::move(l));
        }

        // Retrieve the next value from the queue (pop value from front of
        // queue). This method blocks if the value queue is empty. Waiting
        // threads are resumed automatically as soon as new values are placed
        // into the value queue.
        ValueType get_value()
        {
            std::unique_lock<mutex_type> l(mtx_);

            // cond_.wait() unlocks the lock before suspension and re-locks it
            // afterwards. During this time span another thread may retrieve
            // the next items from the queue for which the thread was resumed.
            while (queue_.empty())
            {
                cond_.wait(l, "queue::get_value");
            }
            HPX_ASSERT(!queue_.empty());

            // get the first value from the value queue and return it to the
            // caller
            ValueType value = std::move(queue_.front());
            queue_.pop();

            return value;
        }

        ValueType get_value(error_code& ec)
        {
            std::unique_lock<mutex_type> l(mtx_);

            // cond_.wait() unlocks the lock before suspension and re-locks it
            // afterwards. During this time span another thread may retrieve
            // the next items from the queue for which the thread was resumed.
            while (queue_.empty())
            {
                cond_.wait(l, "queue::get_value", ec);
                if (ec) return ValueType();
            }
            HPX_ASSERT(!queue_.empty());

            // get the first value from the value queue and return it to the
            // caller
            ValueType value = std::move(queue_.front());
            queue_.pop();

            if (&ec != &throws)
                ec = make_success_code();

            return value;
        }

    private:
        mutex_type mtx_;
        queue_type queue_;
        local::detail::condition_variable cond_;
    };
}}}

#define HPX_REGISTER_QUEUE_DECLARATION(...)                                   \
    HPX_REGISTER_QUEUE_DECLARATION_(__VA_ARGS__)                              \
/**/
#define HPX_REGISTER_QUEUE_DECLARATION_(...)                                  \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_QUEUE_DECLARATION_, HPX_UTIL_PP_NARG(__VA_ARGS__)        \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_QUEUE_DECLARATION_1(type)                                \
    HPX_REGISTER_QUEUE_DECLARATION_2(type, type)                              \
/**/
#define HPX_REGISTER_QUEUE_DECLARATION_2(type, name)                          \
    typedef ::hpx::lcos::server::queue<type>                                  \
        BOOST_PP_CAT(__queue_, BOOST_PP_CAT(type, name));                     \
/**/

#define HPX_REGISTER_QUEUE(...)                                               \
    HPX_REGISTER_QUEUE_(__VA_ARGS__)                                          \
/**/
#define HPX_REGISTER_QUEUE_(...)                                              \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_QUEUE_, HPX_UTIL_PP_NARG(__VA_ARGS__)                    \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_QUEUE_1(type)                                            \
    HPX_REGISTER_QUEUE_2(type, type)                                          \
/**/
#define HPX_REGISTER_QUEUE_2(type, name)                                      \
    typedef ::hpx::lcos::server::queue<type>                                  \
        BOOST_PP_CAT(__queue_, BOOST_PP_CAT(type, name));                     \
    typedef ::hpx::components::managed_component<                             \
            BOOST_PP_CAT(__queue_, BOOST_PP_CAT(type, name))                  \
        > BOOST_PP_CAT(__queue_component_, name);                             \
    HPX_REGISTER_DERIVED_COMPONENT_FACTORY(                                   \
        BOOST_PP_CAT(__queue_component_, name),                               \
        BOOST_PP_CAT(__queue_component_, name),                               \
        BOOST_PP_STRINGIZE(BOOST_PP_CAT(__base_lco_with_value_queue_, name))) \
/**/

#endif
#endif

