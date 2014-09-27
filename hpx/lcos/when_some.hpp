//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/when_some.hpp

#if defined(DOXYGEN)
namespace hpx
{
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
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to when_some.
    ///           - future<vector<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type.
    ///
    /// \note Calling this version of \a when_some where first == last, returns
    ///       a future with an empty vector that is immediately ready.
    ///       Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_some will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename InputIter>
    future<vector<future<typename std::iterator_traits<InputIter>::value_type>>>
    when_some(std::size_t n, Iterator first, Iterator last, error_code& ec = throws);

    /// The function \a when_some is an operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing.
    ///
    /// \param n        [in] The number of futures out of the arguments which
    ///                 have to become ready in order for the returned future
    ///                 to get ready.
    /// \param futures  [in] A vector holding an arbitrary amount of \a future
    ///                 or \a shared_future objects for which \a when_some
    ///                 should wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \note The future returned by the function \a when_some becomes ready
    ///       when at least \a n argument futures have become ready.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to when_some.
    ///           - future<vector<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type.
    ///
    /// \note Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_some will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename R>
    future<std::vector<future<R>>>
    when_some(std::size_t n, std::vector<future<R>>&& futures,
        error_code& ec = throws);

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
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \note The future returned by the function \a when_some becomes ready
    ///       when at least \a n argument futures have become ready.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to when_some.
    ///           - future<tuple<future<R0>, future<R1>, future<R2>...>>: If
    ///             inputs are fixed in number and are of heterogeneous types.
    ///             The inputs can be any arbitrary number of future objects.
    ///
    /// \note Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_some will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename ...T>
    future<tuple<future<T>...>>
    when_some(std::size_t n, T &&... futures, error_code& ec = throws);

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
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to when_some_n.
    ///           - future<vector<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type.
    ///
    /// \note Calling this version of \a when_some_n where count == 0, returns
    ///       a future with the same elements as the arguments that is
    ///       immediately ready. Possibly none of the futures in that vector
    ///       are ready.
    ///       Each future and shared_future is waited upon and then copied into
    ///       the collection of the output (returned) future, maintaining the
    ///       order of the futures in the input collection.
    ///       The future returned by \a when_some_n will not throw an exception,
    ///       but the futures held in the output collection may.
    template <typename InputIter>
    future<vector<future<typename std::iterator_traits<InputIter>::value_type>>>
    when_some_n(std::size_t n, Iterator first, std::size_t count,
        error_code& ec = throws);
}
#else

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_WHEN_SOME_APR_19_2012_0203PM)
#define HPX_LCOS_WHEN_SOME_APR_19_2012_0203PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/atomic.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/is_sequence.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/utility/enable_if.hpp>

#include <algorithm>
#include <iterator>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <typename Future>
        struct when_acquire_future
        {
            typedef Future result_type;

            template <typename R>
            BOOST_FORCEINLINE hpx::future<R>
            operator()(hpx::future<R>& future) const
            {
                return std::move(future);
            }

            template <typename R>
            BOOST_FORCEINLINE hpx::shared_future<R>
            operator()(hpx::shared_future<R>& future) const
            {
                return future;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Sequence>
        struct when_some;

        template <typename Sequence, typename Callback>
        struct set_when_on_completed_callback_impl
        {
            explicit set_when_on_completed_callback_impl(
                    when_some<Sequence>& when, Callback const& callback)
              : when_(when),
                callback_(callback)
            {}

            template <typename Future>
            void operator()(Future& future) const
            {
                std::size_t counter = when_.count_.load(boost::memory_order_seq_cst);
                if (counter < when_.needed_count_ && !future.is_ready()) {
                    // handle future only if not enough futures are ready yet
                    // also, do not touch any futures which are already ready

                    typedef
                        typename lcos::detail::shared_state_ptr_for<Future>::type
                        shared_state_ptr;

                    shared_state_ptr const& shared_state =
                        lcos::detail::get_shared_state(future);

                    shared_state->set_on_completed(Callback(callback_));
                }
                else {
                    if (when_.count_.fetch_add(1) + 1 == when_.needed_count_)
                    {
                        when_.goal_reached_on_calling_thread_ = true;
                    }
                }
            }

            template <typename Sequence_>
            void apply(Sequence_& sequence, typename boost::enable_if<
                boost::fusion::traits::is_sequence<Sequence_> >::type* = 0) const
            {
                boost::fusion::for_each(sequence, *this);
            }

            template <typename Sequence_>
            void apply(Sequence_& sequence, typename boost::disable_if<
                boost::fusion::traits::is_sequence<Sequence_> >::type* = 0) const
            {
                std::for_each(sequence.begin(), sequence.end(), *this);
            }

            when_some<Sequence>& when_;
            Callback const& callback_;
        };

        template <typename Sequence, typename Callback>
        void set_on_completed_callback(when_some<Sequence>& when,
            Callback const& callback)
        {
            set_when_on_completed_callback_impl<Sequence, Callback>
                set_on_completed_callback_helper(when, callback);
            set_on_completed_callback_helper.apply(when.lazy_values_);
        }

        template <typename Sequence>
        struct when_some : boost::enable_shared_from_this<when_some<Sequence> > //-V690
        {
        private:
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
            when_some();
            when_some(when_some const&);

        public:
            typedef Sequence argument_type;
            typedef Sequence result_type;

            when_some(argument_type && lazy_values, std::size_t n)
              : lazy_values_(std::move(lazy_values))
              , count_(0)
              , needed_count_(n)
              , goal_reached_on_calling_thread_(false)
            {}

            result_type operator()()
            {
                // set callback functions to executed when future is ready
                set_on_completed_callback(*this,
                    util::bind(
                        &when_some::on_future_ready, this->shared_from_this(),
                        threads::get_self_id()));

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

            result_type lazy_values_;
            boost::atomic<std::size_t> count_;
            std::size_t needed_count_;
            bool goal_reached_on_calling_thread_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    lcos::future<std::vector<Future> >
    when_some(std::size_t n,
        std::vector<Future>& lazy_values,
        error_code& ec = throws)
    {
        BOOST_STATIC_ASSERT_MSG(
            traits::is_future<Future>::value, "invalid use of when_some");

        typedef std::vector<Future> result_type;

        if (n == 0)
        {
            return lcos::make_ready_future(std::move(lazy_values));
        }

        if (n > lazy_values.size())
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::when_some",
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(result_type());
        }

        result_type lazy_values_;
        lazy_values_.reserve(lazy_values.size());
        std::transform(lazy_values.begin(), lazy_values.end(),
            std::back_inserter(lazy_values_),
            detail::when_acquire_future<Future>());

        boost::shared_ptr<detail::when_some<result_type> > f =
            boost::make_shared<detail::when_some<result_type> >(
                std::move(lazy_values_), n);

        lcos::local::futures_factory<result_type()> p(
            util::bind(&detail::when_some<result_type>::operator(), f));

        p.apply();
        return p.get_future();
    }

    template <typename Future>
    lcos::future<std::vector<Future> > //-V659
    when_some(std::size_t n,
        std::vector<Future> && lazy_values,
        error_code& ec = throws)
    {
        return lcos::when_some(n, lazy_values, ec);
    }

    template <typename Iterator>
    lcos::future<std::vector<
        typename lcos::detail::future_iterator_traits<Iterator>::type
    > >
    when_some(std::size_t n, Iterator begin, Iterator end,
        error_code& ec = throws)
    {
        typedef
            typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;
        typedef std::vector<future_type> result_type;

        result_type lazy_values_;
        std::transform(begin, end, std::back_inserter(lazy_values_),
            detail::when_acquire_future<future_type>());

        return lcos::when_some(n, lazy_values_, ec);
    }

    template <typename Iterator>
    lcos::future<std::vector<
        typename lcos::detail::future_iterator_traits<Iterator>::type
    > >
    when_some_n(std::size_t n, Iterator begin, std::size_t count,
        error_code& ec = throws)
    {
        typedef
            typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;
        typedef std::vector<future_type> result_type;

        result_type lazy_values_;
        lazy_values_.reserve(count);

        detail::when_acquire_future<future_type> func;
        for (std::size_t i = 0; i != count; ++i)
            lazy_values_.push_back(func(*begin++));

        return lcos::when_some(n, lazy_values_, ec);
    }

    inline lcos::future<HPX_STD_TUPLE<> >
    when_some(std::size_t n, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<> result_type;

        result_type lazy_values;

        if (n == 0)
        {
            return lcos::make_ready_future(std::move(lazy_values));
        }

        HPX_THROWS_IF(ec, hpx::bad_parameter,
            "hpx::lcos::when_some",
            "number of results to wait for is out of bounds");
        return lcos::make_ready_future(result_type());
    }
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/when_some.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/when_some_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_WAIT_ARGUMENT_LIMIT, <hpx/lcos/when_some.hpp>))               \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

namespace hpx
{
    using lcos::when_some;
    using lcos::when_some_n;
}

#endif

///////////////////////////////////////////////////////////////////////////////
#else // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

#define HPX_WHEN_SOME_DECAY_FUTURE(Z, N, D)                                   \
    typename util::decay<BOOST_PP_CAT(T, N)>::type                            \
    /**/
#define HPX_WHEN_SOME_ACQUIRE_FUTURE(Z, N, D)                                 \
    detail::when_acquire_future<BOOST_PP_CAT(T, N)>()(BOOST_PP_CAT(f, N))     \
    /**/

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    lcos::future<HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WHEN_SOME_DECAY_FUTURE, _)> >
    when_some(std::size_t n, HPX_ENUM_FWD_ARGS(N, T, f),
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            BOOST_PP_ENUM(N, HPX_WHEN_SOME_DECAY_FUTURE, _)>
            result_type;

        result_type lazy_values(BOOST_PP_ENUM(N, HPX_WHEN_SOME_ACQUIRE_FUTURE, _));

        if (n == 0)
        {
            return lcos::make_ready_future(std::move(lazy_values));
        }

        if (n > N)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter,
                "hpx::lcos::when_some",
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(result_type());
        }

        boost::shared_ptr<detail::when_some<result_type> > f =
            boost::make_shared<detail::when_some<result_type> >(
                std::move(lazy_values), n);

        lcos::local::futures_factory<result_type()> p(
            util::bind(&detail::when_some<result_type>::operator(), f));

        p.apply();
        return p.get_future();
    }
}}

#undef HPX_WHEN_SOME_DECAY_FUTURE
#undef HPX_WHEN_SOME_ACQUIRE_FUTURE
#undef N

#endif

#endif
