//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_PACKAGED_TASK_MAR_01_2012_0121PM)
#define HPX_LCOS_LOCAL_PACKAGED_TASK_MAR_01_2012_0121PM

#include <hpx/config.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/promise.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/function.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_void.hpp>
#include <boost/utility/enable_if.hpp>

#include <utility>

namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Result, typename Signature>
        class packaged_task_base
        {
        private:
            HPX_MOVABLE_ONLY(packaged_task_base);

        public:
            // construction and destruction
            packaged_task_base()
              : function_()
              , promise_()
            {}

            explicit packaged_task_base(util::function_nonser<Signature> const& f)
              : function_(f)
              , promise_()
            {}

            explicit packaged_task_base(util::function_nonser<Signature> && f)
              : function_(std::move(f))
              , promise_()
            {}

            packaged_task_base(packaged_task_base && other)
              : function_(std::move(other.function_))
              , promise_(std::move(other.promise_))
            {}

            packaged_task_base& operator=(packaged_task_base && other)
            {
                if (this != &other)
                {
                    function_ = std::move(other.function_);
                    promise_ = std::move(other.promise_);
                }
                return *this;
            }

            void swap(packaged_task_base& other) HPX_NOEXCEPT
            {
                function_.swap(other.function_);
                promise_.swap(other.promise_);
            }

            // synchronous execution
            template <typename ...Ts>
            void invoke(boost::mpl::false_, Ts&&... vs)
            {
                if (function_.empty())
                {
                    HPX_THROW_EXCEPTION(no_state,
                        "packaged_task_base<Signature>::get_future",
                        "this packaged_task has no valid shared state");
                    return;
                }

                try {
                    promise_.set_value(function_(std::forward<Ts>(vs)...));
                }
                catch(...) {
                    promise_.set_exception(boost::current_exception());
                }
            }

            template <typename ...Ts>
            void invoke(boost::mpl::true_, Ts&&... vs)
            {
                if (function_.empty())
                {
                    HPX_THROW_EXCEPTION(no_state,
                        "packaged_task_base<Signature>::get_future",
                        "this packaged_task has no valid shared state");
                    return;
                }

                try {
                    function_(std::forward<Ts>(vs)...);
                    promise_.set_value();
                }
                catch(...) {
                    promise_.set_exception(boost::current_exception());
                }
            }

            // Result retrieval
            lcos::future<Result> get_future(error_code& ec = throws)
            {
                if (function_.empty()) {
                    HPX_THROWS_IF(ec, no_state,
                        "packaged_task_base<Signature>::get_future",
                        "this packaged_task has no valid shared state");
                    return lcos::future<Result>();
                }
                return promise_.get_future();
            }

            bool valid() const HPX_NOEXCEPT
            {
                return !function_.empty() && promise_.valid();
            }

            void reset(error_code& ec = throws)
            {
                if (function_.empty()) {
                    HPX_THROWS_IF(ec, no_state,
                        "packaged_task_base<Signature>::get_future",
                        "this packaged_task has no valid shared state");
                    return;
                }
                promise_ = local::promise<Result>();
            }

        protected:
            util::function_nonser<Signature> function_;

        private:
            local::promise<Result> promise_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig>
    class packaged_task;

    template <typename R, typename ...Ts>
    class packaged_task<R(Ts...)>
      : private detail::packaged_task_base<R, R(Ts...)>
    {
        HPX_MOVABLE_ONLY(packaged_task);

        typedef detail::packaged_task_base<R, R(Ts...)> base_type;

    public:
        // support for result_of
        typedef R result_type;

        // construction and destruction
        packaged_task()
          : base_type()
        {}

        template <typename F>
        explicit packaged_task(F && f,
            typename boost::enable_if_c<
                !boost::is_same<typename util::decay<F>::type, packaged_task>::value
             && traits::is_callable<typename util::decay<F>::type(Ts...), R>::value
            >::type* = 0)
          : base_type(std::forward<F>(f))
        {}

        packaged_task(packaged_task && other)
          : base_type(std::move(other))
        {}

        packaged_task& operator=(packaged_task && rhs)
        {
            base_type::operator=(std::move(rhs));
            return *this;
        }

        void swap(packaged_task& other) HPX_NOEXCEPT
        {
            base_type::swap(other);
        }

        template <typename ... Vs>
        void operator()(Vs&&... vs)
        {
            base_type::invoke(boost::is_void<R>(), std::forward<Vs>(vs)...);
        }

        // result retrieval
        using base_type::get_future;
        using base_type::valid;
        using base_type::reset;
    };
}}}

#endif
