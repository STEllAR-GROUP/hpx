//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/wait_some.hpp

#if !defined(HPX_LCOS_WAIT_SOME_APR_19_2012_0203PM)
#define HPX_LCOS_WAIT_SOME_APR_19_2012_0203PM

#if defined(DOXYGEN)
namespace hpx
{
    /// The function \a wait_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_all should wait.
    /// \param last     [in] The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_all should wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \note The future returned by the function \a wait_some becomes ready
    ///       when at least \a n argument futures have become ready.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to wait_some.
    ///           - future<vector<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type.
    ///
    /// \note Calling this version of \a wait_some where first == last, returns
    ///       a future with an empty vector that is immediately ready.
    ///       Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a wait_some will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename InputIter>
    future<vector<future<typename std::iterator_traits<InputIter>::value_type>>>
    wait_some(std::size_t n, Iterator first, Iterator last, error_code& ec = throws);

    /// The function \a wait_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param futures  [in] A vector holding an arbitrary amount of \a future
    ///                 or \a shared_future objects for which \a wait_some
    ///                 should wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \note The function \a wait_all returns after \a n futures have become
    ///       ready. All input futures are still valid after \a wait_all
    ///       returns.
    ///
    /// \note Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a wait_some will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename R>
    void wait_some(std::size_t n, std::vector<future<R>>&& futures,
        error_code& ec = throws);

    /// The function \a wait_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param futures  [in] An array holding an arbitrary amount of \a future
    ///                 or \a shared_future objects for which \a wait_some
    ///                 should wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \note The function \a wait_all returns after \a n futures have become
    ///       ready. All input futures are still valid after \a wait_all
    ///       returns.
    ///
    /// \note Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a wait_some will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename R, std::size_t N>
    void wait_some(std::size_t n, std::array<future<R>, N>&& futures,
        error_code& ec = throws);

    /// The function \a wait_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param futures  [in] An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a wait_some should wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \note The function \a wait_all returns after \a n futures have become
    ///       ready. All input futures are still valid after \a wait_all
    ///       returns.
    ///
    /// \note Calling this version of \a wait_some where first == last, returns
    ///       a future with an empty vector that is immediately ready.
    ///       Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a wait_some will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename ...T>
    void wait_some(std::size_t n, T &&... futures, error_code& ec = throws);

    /// The function \a wait_some_n is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_all should wait.
    /// \param count    [in] The number of elements in the sequence starting at
    ///                 \a first.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \note The function \a wait_all returns after \a n futures have become
    ///       ready. All input futures are still valid after \a wait_all
    ///       returns.
    ///
    /// \return This function returns an Iterator referring to the first
    ///         element after the last processed input element.
    ///
    /// \note Calling this version of \a wait_some_n where count == 0, returns
    ///       a future with the same elements as the arguments that is
    ///       immediately ready. Possibly none of the futures in that vector
    ///       are ready.
    ///       Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a wait_some_n will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename InputIter>
    InputIter wait_some_n(std::size_t n, Iterator first,
        std::size_t count, error_code& ec = throws);
}
#else

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/acquire_shared_state.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/atomic.hpp>

#include <algorithm>
#if defined(HPX_HAVE_CXX11_STD_ARRAY)
#include <array>
#endif
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Sequence>
        struct wait_some;

        template <typename Sequence>
        struct set_wait_some_callback_impl
        {
            explicit set_wait_some_callback_impl(wait_some<Sequence>& wait)
              : wait_(wait)
            {}

            template <typename SharedState>
            void operator()(SharedState& shared_state,
                typename std::enable_if<
                    traits::is_shared_state<SharedState>::value
                >::type* = nullptr) const
            {
                std::size_t counter =
                    wait_.count_.load(boost::memory_order_seq_cst);
                if (counter < wait_.needed_count_ &&
                    shared_state.get() != nullptr && !shared_state->is_ready())
                {
                    // handle future only if not enough futures are ready yet
                    // also, do not touch any futures which are already ready

                    shared_state->execute_deferred();

                    // execute_deferred might have made the future ready
                    if (!shared_state->is_ready())
                    {
                        shared_state->set_on_completed(
                            util::deferred_call(
                                &wait_some<Sequence>::on_future_ready,
                                wait_.shared_from_this(),
                                threads::get_self_id()));
                        return;
                    }
                }
                if (wait_.count_.fetch_add(1) + 1 == wait_.needed_count_)
                {
                    wait_.goal_reached_on_calling_thread_ = true;
                }
            }

            template <typename Sequence_>
            HPX_FORCEINLINE
            void operator()(Sequence_& sequence,
                typename std::enable_if<
                    !traits::is_shared_state<Sequence_>::value
                >::type* = nullptr) const
            {
                apply(sequence);
            }

            template <typename Tuple, std::size_t ...Is>
            HPX_FORCEINLINE
            void apply(Tuple& tuple, util::detail::pack_c<std::size_t, Is...>) const
            {
                int const _sequencer[]= {
                    (((*this)(util::get<Is>(tuple))), 0)...
                };
                (void)_sequencer;
            }

            template <typename ...Ts>
            HPX_FORCEINLINE
            void apply(util::tuple<Ts...>& sequence) const
            {
                apply(sequence,
                    typename util::detail::make_index_pack<sizeof...(Ts)>::type());
            }

            template <typename Sequence_>
            HPX_FORCEINLINE
            void apply(Sequence_& sequence) const
            {
                std::for_each(sequence.begin(), sequence.end(), *this);
            }

            wait_some<Sequence>& wait_;
        };

        template <typename Sequence>
        void set_on_completed_callback(wait_some<Sequence>& wait)
        {
            set_wait_some_callback_impl<Sequence> callback(wait);
            callback.apply(wait.lazy_values_);
        }

        template <typename Sequence>
        struct wait_some : std::enable_shared_from_this<wait_some<Sequence> > //-V690
        {
        public:
            void on_future_ready(threads::thread_id_type const& id)
            {
                if (count_.fetch_add(1) + 1 == needed_count_)
                {
                    // reactivate waiting thread only if it's not us
                    if (id != threads::get_self_id())
                        threads::set_thread_state(id, threads::pending);
                    else
                        goal_reached_on_calling_thread_ = true;
                }
            }

        private:
            // workaround gcc regression wrongly instantiating constructors
            wait_some();
            wait_some(wait_some const&);

        public:
            typedef Sequence argument_type;

            wait_some(argument_type && lazy_values, std::size_t n)
              : lazy_values_(std::move(lazy_values))
              , count_(0)
              , needed_count_(n)
              , goal_reached_on_calling_thread_(false)
            {}

            void operator()()
            {
                // set callback functions to executed wait future is ready
                set_on_completed_callback(*this);

                // if all of the requested futures are already set, our
                // callback above has already been called often enough, otherwise
                // we suspend ourselves
                if (!goal_reached_on_calling_thread_)
                {
                    // wait for any of the futures to return to become ready
                    this_thread::suspend(threads::suspended,
                        "hpx::detail::wait_some::operator()");
                }

                // at least N futures should be ready
                HPX_ASSERT(count_.load(boost::memory_order_seq_cst) >= needed_count_);
            }

            argument_type lazy_values_;
            boost::atomic<std::size_t> count_;
            std::size_t const needed_count_;
            bool goal_reached_on_calling_thread_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    void wait_some(std::size_t n,
        std::vector<Future> const& lazy_values,
        error_code& ec = throws)
    {
        static_assert(
            traits::is_future<Future>::value, "invalid use of wait_some");

        typedef
            typename traits::detail::shared_state_ptr_for<Future>::type
            shared_state_ptr;
        typedef std::vector<shared_state_ptr> result_type;

        if (n == 0)
        {
            return;
        }

        if (n > lazy_values.size())
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }

        result_type lazy_values_;
        std::transform(lazy_values.begin(), lazy_values.end(),
            std::back_inserter(lazy_values_),
            traits::detail::wait_get_shared_state<Future>());

        std::shared_ptr<detail::wait_some<result_type> > f =
            std::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);

        return (*f.get())();
    }

    template <typename Future>
    void wait_some(std::size_t n,
        std::vector<Future>& lazy_values,
        error_code& ec = throws)
    {
        return lcos::wait_some(
            n, const_cast<std::vector<Future> const&>(lazy_values), ec);
    }

    template <typename Future>
    void wait_some(std::size_t n,
        std::vector<Future> && lazy_values,
        error_code& ec = throws)
    {
        return lcos::wait_some(
            n, const_cast<std::vector<Future> const&>(lazy_values), ec);
    }

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, std::size_t N>
    void wait_some(std::size_t n,
        std::array<Future, N> const& lazy_values,
        error_code& ec = throws)
    {
        static_assert(
            traits::is_future<Future>::value, "invalid use of wait_some");

        typedef
            typename traits::detail::shared_state_ptr_for<Future>::type
            shared_state_ptr;
        typedef std::array<shared_state_ptr, N> result_type;

        if (n == 0)
        {
            return;
        }

        if (n > lazy_values.size())
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }

        result_type lazy_values_;
        std::transform(lazy_values.begin(), lazy_values.end(),
            lazy_values_.begin(),
            traits::detail::wait_get_shared_state<Future>());

        std::shared_ptr<detail::wait_some<result_type> > f =
            std::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);

        return (*f.get())();
    }

    template <typename Future, std::size_t N>
    void wait_some(std::size_t n,
        std::array<Future, N>& lazy_values,
        error_code& ec = throws)
    {
        return lcos::wait_some(
            n, const_cast<std::array<Future, N> const&>(lazy_values), ec);
    }

    template <typename Future, std::size_t N>
    void wait_some(std::size_t n,
        std::array<Future, N> && lazy_values,
        error_code& ec = throws)
    {
        return lcos::wait_some(
            n, const_cast<std::array<Future, N> const&>(lazy_values), ec);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    typename util::always_void<
        typename lcos::detail::future_iterator_traits<Iterator>::type
    >::type
    wait_some(std::size_t n, Iterator begin, Iterator end,
        error_code& ec = throws)
    {
        typedef
            typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;
        typedef
            typename traits::detail::shared_state_ptr_for<future_type>::type
            shared_state_ptr;
        typedef std::vector<shared_state_ptr> result_type;

        result_type lazy_values_;
        std::transform(begin, end, std::back_inserter(lazy_values_),
            traits::detail::wait_get_shared_state<future_type>());

        std::shared_ptr<detail::wait_some<result_type> > f =
            std::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);

        return (*f.get())();
    }

    template <typename Iterator>
    Iterator
    wait_some_n(std::size_t n, Iterator begin,
        std::size_t count, error_code& ec = throws)
    {
        typedef
            typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;
        typedef
            typename traits::detail::shared_state_ptr_for<future_type>::type
            shared_state_ptr;
        typedef std::vector<shared_state_ptr> result_type;

        result_type lazy_values_;
        lazy_values_.resize(count);
        traits::detail::wait_get_shared_state<future_type> func;
        for (std::size_t i = 0; i != count; ++i)
            lazy_values_.push_back(func(*begin++));

        std::shared_ptr<detail::wait_some<result_type> > f =
            std::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);

        (*f.get())();

        return begin;
    }

    inline void wait_some(std::size_t n, error_code& ec = throws)
    {
        if (n == 0)
        {
            return;
        }

        HPX_THROWS_IF(ec, hpx::bad_parameter,
            "hpx::lcos::wait_some",
            "number of results to wait for is out of bounds");
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    void wait_some(std::size_t n, hpx::future<T> && f, error_code& ec = throws)
    {
        if (n != 1)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }

        f.wait();
    }

    template <typename T>
    void wait_some(std::size_t n, hpx::shared_future<T> && f, error_code& ec = throws)
    {
        if (n != 1)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }

        f.wait();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    void wait_some(std::size_t n, error_code& ec, Ts&&...ts)
    {
        typedef util::tuple<
                typename traits::detail::shared_state_ptr_for<Ts>::type...
            > result_type;

        result_type lazy_values_ =
            result_type(traits::detail::get_shared_state(ts)...);

        if (n == 0)
        {
            return;
        }

        if (n > sizeof...(Ts))
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }

        std::shared_ptr<detail::wait_some<result_type> > f =
            std::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);

        return (*f.get())();
    }

    template <typename... Ts>
    void wait_some(std::size_t n, Ts&&...ts)
    {
        typedef util::tuple<
                typename traits::detail::shared_state_ptr_for<Ts>::type...
            > result_type;

        result_type lazy_values_ =
            result_type(traits::detail::get_shared_state(ts)...);

        if (n == 0)
        {
            return;
        }

        if (n > sizeof...(Ts))
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "hpx::lcos::wait_some",
                "number of results to wait for is out of bounds");
            return;
        }

        std::shared_ptr<detail::wait_some<result_type> > f =
            std::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), n);

        return (*f.get())();
    }
}}

namespace hpx
{
    using lcos::wait_some;
    using lcos::wait_some_n;
}

#endif // DOXYGEN
#endif
