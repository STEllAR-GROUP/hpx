//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_INVOKE_NO_MORE_THAN_MAR_30_2014_0616PM)
#define HPX_RUNTIME_ACTIONS_INVOKE_NO_MORE_THAN_MAR_30_2014_0616PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/counting_semaphore.hpp>
#include <hpx/util/static.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/action_decorate_function.hpp>
#include <hpx/traits/action_decorate_continuation.hpp>

#include <memory>

namespace hpx { namespace actions { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Semaphore>
    struct signal_on_exit
    {
        signal_on_exit(Semaphore& sem)
            : sem_(sem)
        {
            sem_.wait();
        }

        ~signal_on_exit()
        {
            sem_.signal();
        }

    private:
        Semaphore& sem_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, int N>
    struct action_decorate_function_semaphore
    {
        typedef lcos::local::counting_semaphore_var<
                hpx::lcos::local::spinlock, N
            > semaphore_type;

        struct tag {};

        static semaphore_type& get_sem()
        {
            util::static_<semaphore_type, tag> sem;
            return sem.get();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, int N>
    struct action_decorate_function
    {
        // This wrapper is needed to stop infinite recursion when
        // trying to get the possible additional function decoration
        // from the component
        struct action_wrapper
        {
            typedef typename Action::component_type component_type;
        };

        static_assert(
            !Action::direct_execution::value,
            "explicit decoration of direct actions is not supported");

        typedef action_decorate_function_semaphore<Action, N>
            construct_semaphore_type;

        // If the action returns something which is not a future, we inject
        // a semaphore into the call graph.
        static threads::thread_state_enum thread_function(
            threads::thread_state_ex_enum state,
            threads::thread_function_type f)
        {
            typedef typename construct_semaphore_type::semaphore_type
                semaphore_type;

            signal_on_exit<semaphore_type> on_exit(
                construct_semaphore_type::get_sem());
            return f(state);
        }

        template <typename F>
        static threads::thread_function_type
        call(naming::address::address_type lva, F && f, boost::mpl::false_)
        {
            return util::bind(
                util::one_shot(&action_decorate_function::thread_function),
                util::placeholders::_1,
                traits::action_decorate_function<action_wrapper>::call(
                    lva, std::forward<F>(f))
            );
        }

        // If the action returns a future we wait on the semaphore as well,
        // however it will be signaled once the future gets ready only (see
        // the traits::action_schedule_thread<> below).
        static threads::thread_state_enum thread_function_future(
            threads::thread_state_ex_enum state,
            threads::thread_function_type f)
        {
            construct_semaphore_type::get_sem().wait();
            return f(state);
        }

        template <typename F>
        static threads::thread_function_type
        call(naming::address::address_type lva, F && f, boost::mpl::true_)
        {
            return util::bind(
                util::one_shot(&action_decorate_function::thread_function_future),
                util::placeholders::_1,
                traits::action_decorate_function<action_wrapper>::call(
                    lva, std::forward<F>(f))
            );
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F>
        static threads::thread_function_type
        call(naming::address::address_type lva, F&& f)
        {
            typedef typename Action::result_type result_type;
            typedef typename traits::is_future<result_type>::type is_future;
            return call(lva, std::forward<F>(f), is_future());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, int N>
    class wrapped_continuation
      : public typed_continuation<typename Action::result_type>
    {
        typedef action_decorate_function_semaphore<Action, N>
            construct_semaphore_type;
        typedef typename Action::result_type result_type;
        typedef typed_continuation<result_type> base_type;

    public:
        wrapped_continuation(std::unique_ptr<continuation> cont)
          : cont_(std::move(cont))
        {}

        wrapped_continuation(wrapped_continuation && o)
          : cont_(std::move(o.cont_))
        {}

        void trigger()
        {
            if (cont_) cont_->trigger();
            construct_semaphore_type::get_sem().signal();
        }

        void deferred_trigger(result_type&& result)
        {
            if (cont_) {
                static_cast<base_type *>(cont_.get())->
                    deferred_trigger(std::move(result));
            }
            construct_semaphore_type::get_sem().signal();
        }

        void trigger_value(result_type && result)
        {
            // if the future is ready, send the result back immediately
            if (result.is_ready()) {
                deferred_trigger(std::move(result));
                return;
            }

            // attach continuation to this future which will send the result back
            // once its ready
            result.then(
                util::bind(&wrapped_continuation::deferred_trigger,
                    std::move(*this),
                    util::placeholders::_1));
        }

        void trigger_error(boost::exception_ptr const& e)
        {
            if (cont_) cont_->trigger_error(e);
            construct_semaphore_type::get_sem().signal();
        }
        void trigger_error(boost::exception_ptr && e)
        {
            if (cont_) cont_->trigger_error(std::move(e));
            construct_semaphore_type::get_sem().signal();
        }

    private:
        std::unique_ptr<continuation> cont_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, int N>
    struct action_decorate_continuation
    {
        static_assert(
            !Action::direct_execution::value,
            "explicit decoration of direct actions is not supported");

        typedef action_decorate_function_semaphore<Action, N>
            construct_semaphore_type;

        ///////////////////////////////////////////////////////////////////////
        // If the action returns something which is not a future, we do nothing
        // special.
        template <typename Continuation>
        static bool call(Continuation& cont, boost::mpl::false_)
        {
            return false;
        }

        // If the action returns a future we wrap the given continuation to
        // be able to signal the semaphore after the wrapped action has
        // returned.
        template <typename Continuation>
        static bool call(Continuation& c, boost::mpl::true_)
        {
            c = std::unique_ptr<continuation>(
                new wrapped_continuation<Action, N>(std::move(c)));
            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        static bool call(std::unique_ptr<continuation>& cont)
        {
            typedef typename Action::result_type result_type;
            typedef typename traits::is_future<result_type>::type is_future;
            return call(cont, is_future());
        }
    };
}}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_INVOKE_NO_MORE_THAN(action, maxnum)                        \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_decorate_function<action>                               \
          : hpx::actions::detail::action_decorate_function<action, maxnum>    \
        {};                                                                   \
                                                                              \
        template <>                                                           \
        struct action_decorate_continuation<action>                           \
          : hpx::actions::detail::action_decorate_continuation<action, maxnum>\
        {};                                                                   \
    }}                                                                        \
/**/

#endif
