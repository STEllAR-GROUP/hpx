//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_FUTURE_WAIT_OCT_23_2008_1140AM)
#define HPX_LCOS_FUTURE_WAIT_OCT_23_2008_1140AM

#include <hpx/hpx_fwd.hpp>

#include <hpx/traits/acquire_shared_state.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/wait_all.hpp>

#include <boost/atomic.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/type_traits/is_void.hpp>
#include <boost/utility/enable_if.hpp>

#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <typename Future>
        struct wait_acquire_future
        {
            typedef Future result_type;

            template <typename R>
            HPX_FORCEINLINE hpx::future<R>
            operator()(hpx::future<R>& future) const
            {
                return std::move(future);
            }

            template <typename R>
            HPX_FORCEINLINE hpx::shared_future<R>
            operator()(hpx::shared_future<R>& future) const
            {
                return future;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // This version has a callback to be invoked for each future when it
        // gets ready.
        template <typename Future, typename F>
        struct wait_each
        {
        private:
            HPX_MOVABLE_ONLY(wait_each);

        protected:
            void on_future_ready_(threads::thread_id_type const& id)
            {
                std::size_t oldcount = ready_count_.fetch_add(1);
                HPX_ASSERT(oldcount < lazy_values_.size());

                if (oldcount + 1 == lazy_values_.size())
                {
                    // reactivate waiting thread only if it's not us
                    if (id != threads::get_self_id())
                        threads::set_thread_state(id, threads::pending);
                    else
                        goal_reached_on_calling_thread_ = true;
                }
            }

            template <typename Index>
            void on_future_ready_(Index i, threads::thread_id_type const& id,
                boost::mpl::false_)
            {
                if (lazy_values_[i].has_value()) {
                    if (success_counter_)
                        ++*success_counter_;
                    // invoke callback function
                    f_(i, lazy_values_[i].get());
                }

                // keep track of ready futures
                on_future_ready_(id);
            }

            template <typename Index>
            void on_future_ready_(Index i, threads::thread_id_type const& id,
                boost::mpl::true_)
            {
                if (lazy_values_[i].has_value()) {
                    if (success_counter_)
                        ++*success_counter_;
                    // invoke callback function
                    f_(i);
                }

                // keep track of ready futures
                on_future_ready_(id);
            }

            void on_future_ready(std::size_t i, threads::thread_id_type const& id)
            {
                on_future_ready_(i, id,
                    boost::is_void<typename traits::future_traits<Future>::type>());
            }

        public:
            typedef std::vector<Future> argument_type;
            typedef std::vector<Future> result_type;

            template <typename F_>
            wait_each(argument_type const& lazy_values, F_ && f,
                    boost::atomic<std::size_t>* success_counter)
              : lazy_values_(lazy_values),
                ready_count_(0),
                f_(std::forward<F>(f)),
                success_counter_(success_counter)
            {}

            template <typename F_>
            wait_each(argument_type && lazy_values, F_ && f,
                    boost::atomic<std::size_t>* success_counter)
              : lazy_values_(std::move(lazy_values)),
                ready_count_(0),
                f_(std::forward<F>(f)),
                success_counter_(success_counter)
            {}

            wait_each(wait_each && rhs)
              : lazy_values_(std::move(rhs.lazy_values_)),
                ready_count_(rhs.ready_count_.load()),
                f_(std::move(rhs.f_)),
                success_counter_(rhs.success_counter_)
            {
                rhs.success_counter_ = 0;
            }

            wait_each& operator= (wait_each && rhs)
            {
                if (this != &rhs) {
                    lazy_values_ = std::move(rhs.lazy_values_);
                    ready_count_.store(rhs.ready_count_.load());
                    rhs.ready_count_ = 0;
                    f_ = std::move(rhs.f_);
                    success_counter_ = rhs.success_counter_;
                    rhs.success_counter_ = 0;
                }
                return *this;
            }

            result_type operator()()
            {
                ready_count_.store(0);
                goal_reached_on_calling_thread_ = false;

                // set callback functions to executed when future is ready
                std::size_t size = lazy_values_.size();
                threads::thread_id_type id = threads::get_self_id();
                for (std::size_t i = 0; i != size; ++i)
                {
                    typedef
                        typename traits::detail::shared_state_ptr_for<Future>::type
                        shared_state_ptr;
                    shared_state_ptr current =
                        traits::detail::get_shared_state(lazy_values_[i]);

                    current->execute_deferred();
                    current->set_on_completed(
                        util::bind(&wait_each::on_future_ready, this, i, id));
                }

                // If all of the requested futures are already set then our
                // callback above has already been called, otherwise we suspend
                // ourselves.
                if (!goal_reached_on_calling_thread_)
                {
                    // wait for all of the futures to return to become ready
                    this_thread::suspend(threads::suspended,
                        "hpx::lcos::detail::wait_each::operator()");
                }

                // all futures should be ready
                HPX_ASSERT(ready_count_ == size);

                return std::move(lazy_values_);
            }

            std::vector<Future> lazy_values_;
            boost::atomic<std::size_t> ready_count_;
            typename boost::remove_reference<F>::type f_;
            boost::atomic<std::size_t>* success_counter_;
            bool goal_reached_on_calling_thread_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronous versions.

    /// The one argument version is special in the sense that it returns the
    /// expected value directly (without wrapping it into a tuple).
    template <typename Future, typename F>
    inline typename boost::enable_if_c<
        !boost::is_void<typename traits::future_traits<Future>::type>::value
      , std::size_t
    >::type
    wait(Future && f1, F && f)
    {
        f(0, f1.get());
        return 1;
    }

    template <typename Future, typename F>
    inline typename boost::enable_if_c< //-V659
        boost::is_void<typename traits::future_traits<Future>::type>::value
      , std::size_t
    >::type
    wait(Future && f1, F && f)
    {
        f1.get();
        f(0);
        return 1;
    }

    //////////////////////////////////////////////////////////////////////////
    // This overload of wait() will make sure that the passed function will be
    // invoked as soon as a value becomes available, it will not wait for all
    // results to be there.
    template <typename Future, typename F>
    inline std::size_t
    wait(std::vector<Future>& lazy_values, F && f,
        boost::int32_t suspend_for = 10)
    {
        typedef std::vector<Future> return_type;

        if (lazy_values.empty())
            return 0;

        return_type lazy_values_;
        lazy_values_.reserve(lazy_values.size());
        std::transform(lazy_values.begin(), lazy_values.end(),
            std::back_inserter(lazy_values_),
            detail::wait_acquire_future<Future>());

        boost::atomic<std::size_t> success_counter(0);
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                detail::wait_each<Future, F>(std::move(lazy_values_),
                    std::forward<F>(f), &success_counter));

        p.apply();
        p.get_future().get();

        return success_counter.load();
    }

    template <typename Future, typename F>
    inline std::size_t
    wait(std::vector<Future> && lazy_values, F && f,
        boost::int32_t suspend_for = 10)
    {
        return wait(lazy_values, std::forward<F>(f), suspend_for);
    }

    template <typename Future, typename F>
    inline std::size_t
    wait(std::vector<Future> const& lazy_values, F && f,
        boost::int32_t suspend_for = 10)
    {
        typedef std::vector<Future> return_type;

        if (lazy_values.empty())
            return 0;

        return_type lazy_values_;
        lazy_values_.reserve(lazy_values.size());
        std::transform(lazy_values.begin(), lazy_values.end(),
            std::back_inserter(lazy_values_),
            detail::wait_acquire_future<Future>());

        boost::atomic<std::size_t> success_counter(0);
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                detail::wait_each<Future, F>(std::move(lazy_values_),
                    std::forward<F>(f), &success_counter));

        p.apply();
        p.get_future().get();

        return success_counter.load();
    }
}}

#endif
