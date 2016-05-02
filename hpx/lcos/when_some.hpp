//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/when_some.hpp

#if !defined(HPX_LCOS_WHEN_SOME_APR_19_2012_0203PM)
#define HPX_LCOS_WHEN_SOME_APR_19_2012_0203PM

#if defined(DOXYGEN)
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// Result type for \a when_some, contains a sequence of futures and
    /// indices pointing to ready futures.
    template <typename Sequence>
    struct when_some_result
    {
        /// List of indices of futures which became ready
        std::vector<std::size_t> indices;

        ///< The sequence of futures as passed to \a hpx::when_some
        Sequence futures;
    };

    /// The function \a when_some is an operator allowing to join on the result
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
    /// \note The future returned by the function \a when_some becomes ready
    ///       when at least \a n argument futures have become ready.
    ///
    /// \return   Returns a when_some_result holding the same list of futures
    ///           as has been passed to when_some and indices pointing to
    ///           ready futures.
    ///           - future<when_some_result<Container<future<R>>>>: If the input
    ///             cardinality is unknown at compile time and the futures
    ///             are all of the same type. The order of the futures in the
    ///             output container will be the same as given by the input
    ///             iterator.
    ///
    /// \note Calling this version of \a when_some where first == last, returns
    ///       a future with an empty container that is immediately ready.
    ///       Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_some will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename InputIter, typename Container =
        vector<future<typename std::iterator_traits<InputIter>::value_type>>>
    future<when_some_result<Container>>
    when_some(std::size_t n, Iterator first, Iterator last, error_code& ec = throws);

    /// The function \a when_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param futures  [in] A container holding an arbitrary amount of \a future
    ///                 or \a shared_future objects for which \a when_some
    ///                 should wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \note The future returned by the function \a when_some becomes ready
    ///       when at least \a n argument futures have become ready.
    ///
    /// \return   Returns a when_some_result holding the same list of futures
    ///           as has been passed to when_some and indices pointing to
    ///           ready futures.
    ///           - future<when_some_result<Container<future<R>>>>: If the input
    ///             cardinality is unknown at compile time and the futures
    ///             are all of the same type. The order of the futures in the
    ///             output container will be the same as given by the input
    ///             iterator.
    ///
    /// \note Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_some will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename Range>
    future<when_some_result<Range>>
    when_some(std::size_t n, Range&& futures,
        error_code& ec = throws);

    /// The function \a when_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    /// \param futures  [in] An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a when_some should wait.
    ///
    /// \note The future returned by the function \a when_some becomes ready
    ///       when at least \a n argument futures have become ready.
    ///
    /// \return   Returns a when_some_result holding the same list of futures
    ///           as has been passed to when_some and an index pointing to a
    ///           ready future..
    ///           - future<when_some_result<tuple<future<T0>, future<T1>...>>>:
    ///             If inputs are fixed in number and are of heterogeneous
    ///             types. The inputs can be any arbitrary number of future
    ///             objects.
    ///           - future<when_some_result<tuple<>>> if \a when_some is
    ///             called with zero arguments.
    ///             The returned future will be initially ready.
    ///
    /// \note Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_some will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename ...T>
    future<when_some_result<tuple<future<T>...>>>
    when_some(std::size_t n, error_code& ec, T &&... futures);

    /// The function \a when_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param futures  [in] An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a when_some should wait.
    ///
    /// \note The future returned by the function \a when_some becomes ready
    ///       when at least \a n argument futures have become ready.
    ///
    /// \return   Returns a when_some_result holding the same list of futures
    ///           as has been passed to when_some and an index pointing to a
    ///           ready future..
    ///           - future<when_some_result<tuple<future<T0>, future<T1>...>>>:
    ///             If inputs are fixed in number and are of heterogeneous
    ///             types. The inputs can be any arbitrary number of future
    ///             objects.
    ///           - future<when_some_result<tuple<>>> if \a when_some is
    ///             called with zero arguments.
    ///             The returned future will be initially ready.
    ///
    /// \note Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_some will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename ...T>
    future<when_some_result<tuple<future<T>...>>>
    when_some(std::size_t n, T &&... futures);

    /// The function \a when_some_n is an operator allowing to join on the result
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
    /// \note The future returned by the function \a when_some_n becomes ready
    ///       when at least \a n argument futures have become ready.
    ///
    /// \return   Returns a when_some_result holding the same list of futures
    ///           as has been passed to when_some and indices pointing to
    ///           ready futures.
    ///           - future<when_some_result<Container<future<R>>>>: If the input
    ///             cardinality is unknown at compile time and the futures
    ///             are all of the same type. The order of the futures in the
    ///             output container will be the same as given by the input
    ///             iterator.
    ///
    /// \note Calling this version of \a when_some_n where count == 0, returns
    ///       a future with the same elements as the arguments that is
    ///       immediately ready. Possibly none of the futures in that container
    ///       are ready.
    ///       Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_some_n will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename InputIter, typename Container =
        vector<future<typename std::iterator_traits<InputIter>::value_type>>>
    future<when_some_result<Container>>
    when_some_n(std::size_t n, Iterator first, std::size_t count,
        error_code& ec = throws);
}

#else // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/futures_factory.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/traits/acquire_shared_state.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pack.hpp>

#include <boost/atomic.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    template <typename Sequence>
    struct when_some_result
    {
        when_some_result()
          : indices()
          , futures()
        {}

        explicit when_some_result(Sequence&& futures)
          : indices()
          , futures(std::move(futures))
        {}

        when_some_result(when_some_result const& rhs)
          : indices(rhs.indices), futures(rhs.futures)
        {}

        when_some_result(when_some_result&& rhs)
          : indices(std::move(rhs.indices)), futures(std::move(rhs.futures))
        {}

        when_some_result& operator=(when_some_result const& rhs)
        {
            if (this != &rhs)
            {
                indices = rhs.indices;
                futures = rhs.futures;
            }
            return true;
        }

        when_some_result& operator=(when_some_result && rhs)
        {
            if (this != &rhs)
            {
                indices = std::move(rhs.indices);
                futures = std::move(rhs.futures);
            }
            return true;
        }

        std::vector<std::size_t> indices;
        Sequence futures;
    };

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Sequence>
        struct when_some;

        template <typename Sequence>
        struct set_when_some_callback_impl
        {
            explicit set_when_some_callback_impl(when_some<Sequence>& when)
              : when_(when), idx_(0)
            {}

            template <typename Future>
            void operator()(Future& future,
                typename std::enable_if<
                    traits::is_future<Future>::value
                >::type* = 0) const
            {
                std::size_t counter =
                    when_.count_.load(boost::memory_order_seq_cst);
                if (counter < when_.needed_count_) {
                    if (!future.is_ready()) {
                        // handle future only if not enough futures are ready
                        // yet also, do not touch any futures which are already
                        // ready

                        typedef
                            typename traits::detail::shared_state_ptr_for<Future>::type
                            shared_state_ptr;

                        shared_state_ptr const& shared_state =
                            traits::detail::get_shared_state(future);

                        shared_state->execute_deferred();

                        // execute_deferred might have made the future ready
                        if (!shared_state->is_ready())
                        {
                            shared_state->set_on_completed(
                                util::deferred_call(
                                    &when_some<Sequence>::on_future_ready,
                                    when_.shared_from_this(),
                                    idx_, threads::get_self_id()));
                            ++idx_;
                            return;
                        }
                    }

                    when_.lazy_values_.indices.push_back(idx_);
                    if (when_.count_.fetch_add(1) + 1 == when_.needed_count_)
                    {
                        when_.goal_reached_on_calling_thread_ = true;
                    }
                }
                ++idx_;
            }

            template <typename Sequence_>
            HPX_FORCEINLINE
            void operator()(Sequence_& sequence,
                typename std::enable_if<
                    traits::is_future_range<Sequence_>::value
                >::type* = 0) const
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

            when_some<Sequence>& when_;
            mutable std::size_t idx_;
        };

        template <typename Sequence>
        HPX_FORCEINLINE
        void set_on_completed_callback(when_some<Sequence>& when)
        {
            set_when_some_callback_impl<Sequence> callback(when);
            callback.apply(when.lazy_values_.futures);
        }

        template <typename Sequence>
        struct when_some : std::enable_shared_from_this<when_some<Sequence> > //-V690
        {
            typedef lcos::local::spinlock mutex_type;

        public:
            void on_future_ready(std::size_t idx, threads::thread_id_type const& id)
            {
                std::size_t const new_count = count_.fetch_add(1) + 1;
                if (new_count <= needed_count_)
                {
                    {
                        std::lock_guard<mutex_type> l(this->mtx_);
                        lazy_values_.indices.push_back(idx);
                    }
                    if (new_count == needed_count_) {
                        if (id != threads::get_self_id()) {
                            threads::set_thread_state(id, threads::pending);
                        } else {
                            goal_reached_on_calling_thread_ = true;
                        }
                    }
                }
            }

        private:
            // workaround gcc regression wrongly instantiating constructors
            when_some();
            when_some(when_some const&);

        public:
            typedef Sequence argument_type;

            when_some(argument_type && lazy_values, std::size_t n)
              : lazy_values_(std::move(lazy_values))
              , count_(0)
              , needed_count_(n)
              , goal_reached_on_calling_thread_(false)
            {}

            when_some_result<Sequence> operator()()
            {
                // set callback functions to executed when future is ready
                set_on_completed_callback(*this);

                // if all of the requested futures are already set, our
                // callback above has already been called often enough, otherwise
                // we suspend ourselves
                if (!goal_reached_on_calling_thread_)
                {
                    // wait for any of the futures to return to become ready
                    this_thread::suspend(threads::suspended,
                        "hpx::lcos::detail::when_some::operator()");
                }

                // at least N futures should be ready
                HPX_ASSERT(count_.load(boost::memory_order_seq_cst) >= needed_count_);

                return std::move(lazy_values_);
            }

            mutable mutex_type mtx_;
            when_some_result<Sequence> lazy_values_;
            boost::atomic<std::size_t> count_;
            std::size_t needed_count_;
            bool goal_reached_on_calling_thread_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Range>
    typename std::enable_if<traits::is_future_range<Range>::value,
        lcos::future<when_some_result<typename std::decay<Range>::type> > >::type
    when_some(std::size_t n,
        Range& lazy_values,
        error_code& ec = throws)
    {
        typedef Range result_type;

        if (n == 0)
        {
            return lcos::make_ready_future(
                when_some_result<result_type>(std::move(lazy_values)));
        }

        if (n > lazy_values.size())
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::when_some",
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(
                when_some_result<result_type>(result_type()));
        }

        result_type lazy_values_ =
            traits::acquire_future<result_type>()(lazy_values);

        std::shared_ptr<detail::when_some<result_type> > f =
            std::make_shared<detail::when_some<result_type> >(
                std::move(lazy_values_), n);

        lcos::local::futures_factory<when_some_result<result_type>()> p(
            util::deferred_call(&detail::when_some<result_type>::operator(), f));

        p.apply();
        return p.get_future();
    }

    template <typename Range>
    typename std::enable_if<traits::is_future_range<Range>::value,
        lcos::future<when_some_result<typename std::decay<Range>::type> > >::type
    when_some(std::size_t n,
        Range&& lazy_values,
        error_code& ec = throws)
    {
        return lcos::when_some(n, lazy_values, ec);
    }

    template <typename Iterator, typename Container =
        std::vector<typename lcos::detail::future_iterator_traits<Iterator>::type> >
    lcos::future<when_some_result<Container> >
    when_some(std::size_t n, Iterator begin, Iterator end,
        error_code& ec = throws)
    {
        Container lazy_values_;

        typename std::iterator_traits<Iterator>::
            difference_type difference = std::distance(begin, end);
        if (difference > 0)
            traits::detail::reserve_if_vector(
                lazy_values_, static_cast<std::size_t>(difference));

        std::transform(begin, end, std::back_inserter(lazy_values_),
            traits::acquire_future_disp());

        return lcos::when_some(n, lazy_values_, ec);
    }

    template <typename Iterator, typename Container =
        std::vector<typename lcos::detail::future_iterator_traits<Iterator>::type> >
    lcos::future<when_some_result<Container> >
    when_some_n(std::size_t n, Iterator begin, std::size_t count,
        error_code& ec = throws)
    {
        Container lazy_values_;
        traits::detail::reserve_if_vector(lazy_values_, count);

        traits::acquire_future_disp func;
        for (std::size_t i = 0; i != count; ++i)
            lazy_values_.push_back(func(*begin++));

        return lcos::when_some(n, lazy_values_, ec);
    }

    inline lcos::future<when_some_result<util::tuple<> > >
    when_some(std::size_t n, error_code& ec = throws)
    {
        typedef util::tuple<> result_type;

        result_type lazy_values;

        if (n == 0)
        {
            return lcos::make_ready_future(
                when_some_result<result_type>(std::move(lazy_values)));
        }

        HPX_THROWS_IF(ec, hpx::bad_parameter,
            "hpx::lcos::when_some",
            "number of results to wait for is out of bounds");
        return lcos::make_ready_future(when_some_result<result_type>());
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    lcos::future<when_some_result<
        util::tuple<typename traits::acquire_future<Ts>::type...>
    > >
    when_some(std::size_t n, Ts&&... ts)
    {
        typedef util::tuple<
                typename traits::acquire_future<Ts>::type...
            > result_type;

        traits::acquire_future_disp func;
        result_type lazy_values(func(std::forward<Ts>(ts))...);

        if (n == 0)
        {
            return lcos::make_ready_future(
                when_some_result<result_type>(std::move(lazy_values)));
        }

        if (n > sizeof...(Ts))
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "hpx::lcos::when_some",
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(when_some_result<result_type>());
        }

        std::shared_ptr<detail::when_some<result_type> > f =
            std::make_shared<detail::when_some<result_type> >(
                std::move(lazy_values), n);

        lcos::local::futures_factory<when_some_result<result_type>()> p(
            util::deferred_call(&detail::when_some<result_type>::operator(), f));

        p.apply();
        return p.get_future();
    }

    template <typename... Ts>
    lcos::future<when_some_result<
        util::tuple<typename traits::acquire_future<Ts>::type...>
    > >
    when_some(std::size_t n, error_code& ec, Ts&&... ts)
    {
        typedef util::tuple<
                typename traits::acquire_future<Ts>::type...
            > result_type;

        traits::acquire_future_disp func;
        result_type lazy_values(func(std::forward<Ts>(ts))...);

        if (n == 0)
        {
            return lcos::make_ready_future(
                when_some_result<result_type>(std::move(lazy_values)));
        }

        if (n > sizeof...(Ts))
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::when_some",
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(when_some_result<result_type>());
        }

        std::shared_ptr<detail::when_some<result_type> > f =
            std::make_shared<detail::when_some<result_type> >(
                std::move(lazy_values), n);

        lcos::local::futures_factory<when_some_result<result_type>()> p(
            util::deferred_call(&detail::when_some<result_type>::operator(), f));

        p.apply();
        return p.get_future();
    }
}}

namespace hpx
{
    using lcos::when_some_result;
    using lcos::when_some;
    using lcos::when_some_n;
}

#endif // DOXYGEN
#endif
