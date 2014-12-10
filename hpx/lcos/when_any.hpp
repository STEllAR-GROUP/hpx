//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/when_any.hpp

#if defined(DOXYGEN)
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// Result type for \a when_any, contains a sequence of futures and an
    /// index pointing to a ready future.
    template <typename Sequence>
    struct when_any_result
    {
        std::size_t index;
        Sequence futures;
    };

    /// The function \a when_any is a non-deterministic choice operator. It
    /// OR-composes all future objects given and returns a new future object
    /// representing the same list of futures after one future of that list
    /// finishes execution.
    ///
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_any should wait.
    /// \param last     [in] The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_any should wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \return   Returns a when_any_result holding the same list of futures
    ///           as has been passed to when_any and an index pointing to a
    ///           ready future.
    ///           - future<when_any_result<vector<future<R>>>>: If the input
    ///             cardinality is unknown at compile time and the futures
    ///             are all of the same type. The order of the futures in the
    ///             output vector will be the same as given by the input
    ///             iterator.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    template <typename InputIter>
    future<when_any_result<
        vector<future<typename std::iterator_traits<InputIter>::value_type>>>>
    when_any(InputIter first, InputIter last, error_code& ec = throws);

    /// The function \a when_any is a non-deterministic choice operator. It
    /// OR-composes all future objects given and returns a new future object
    /// representing the same list of futures after one future of that list
    /// finishes execution.
    ///
    /// \param futures  [in] A vector holding an arbitrary amount of \a future or
    ///                 \a shared_future objects for which \a when_any should
    ///                 wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \return   Returns a when_any_result holding the same list of futures
    ///           as has been passed to when_any and an index pointing to a
    ///           ready future.
    ///           - future<when_any_result<vector<future<R>>>>: If the input
    ///             cardinality is unknown at compile time and the futures
    ///             are all of the same type. The order of the futures in the
    ///             output vector will be the same as given by the input
    ///             iterator.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    template <typename R>
    future<when_any_result<
        std::vector<future<R>>>>
    when_any(std::vector<future<R>>& futures, error_code& ec = throws);

    /// The function \a when_any is a non-deterministic choice operator. It
    /// OR-composes all future objects given and returns a new future object
    /// representing the same list of futures after one future of that list
    /// finishes execution.
    ///
    /// \param futures  [in] An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a when_any should wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \return   Returns a when_any_result holding the same list of futures
    ///           as has been passed to when_any and an index pointing to a
    ///           ready future..
    ///           - future<when_any_result<tuple<future<T0>, future<T1>...>>>:
    ///             If inputs are fixed in number and are of heterogeneous
    ///             types. The inputs can be any arbitrary number of future
    ///             objects.
    ///           - future<when_any_result<tuple<>>> if \a when_any is called
    ///             with zero arguments.
    ///             The returned future will be initially ready.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    template <typename ...T>
    future<when_any_result<tuple<future<T>...>>>
    when_any(T &&... futures, error_code& ec = throws);

    /// The function \a when_any_n is a non-deterministic choice operator. It
    /// OR-composes all future objects given and returns a new future object
    /// representing the same list of futures after one future of that list
    /// finishes execution.
    ///
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_any_n should wait.
    /// \param count    [in] The number of elements in the sequence starting at
    ///                 \a first.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \return   Returns a when_any_result holding the same list of futures
    ///           as has been passed to when_any and an index pointing to a
    ///           ready future.
    ///           - future<when_any_result<vector<future<R>>>>: If the input
    ///             cardinality is unknown at compile time and the futures
    ///             are all of the same type. The order of the futures in the
    ///             output vector will be the same as given by the input
    ///             iterator.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    ///
    /// \note     None of the futures in the input sequence are invalidated.
    template <typename InputIter>
    future<when_any_result<
        vector<future<typename std::iterator_traits<InputIter>::value_type>>>>
    when_any_n(InputIter first, std::size_t count, error_code& ec = throws);
}
#else

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_WHEN_ANY_APR_17_2012_1143AM)
#define HPX_LCOS_WHEN_ANY_APR_17_2012_1143AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/when_any.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/runtime/threads/thread.hpp>
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
#include <boost/utility/swap.hpp>

#include <algorithm>
#include <iterator>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    template <typename Sequence>
    struct when_any_result
    {
        static std::size_t index_error()
        {
            return static_cast<std::size_t>(-1);
        }

        when_any_result()
          : index(static_cast<size_t>(index_error()))
          , futures()
        {}

        explicit when_any_result(Sequence&& futures)
          : index(index_error())
          , futures(std::move(futures))
        {}

        when_any_result(when_any_result const& rhs)
          : index(rhs.index), futures(rhs.futures)
        {}

        when_any_result(when_any_result&& rhs)
          : index(rhs.index), futures(std::move(rhs.futures))
        {
            rhs.index = index_error();
        }

        std::size_t index;
        Sequence futures;
    };

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Sequence>
        struct when_any;

        template <typename Sequence>
        struct set_when_any_callback_impl
        {
            explicit set_when_any_callback_impl(when_any<Sequence>& when)
              : when_(when), idx_(0)
            {}

            template <typename Future>
            void operator()(Future& future) const
            {
                std::size_t index = when_.index_.load(boost::memory_order_seq_cst);
                if (index == when_any_result<Sequence>::index_error()) {
                    if (!future.is_ready()) {
                        // handle future only if not enough futures are ready yet
                        // also, do not touch any futures which are already ready

                        typedef
                            typename lcos::detail::shared_state_ptr_for<Future>::type
                            shared_state_ptr;

                        shared_state_ptr const& shared_state =
                            lcos::detail::get_shared_state(future);

                        shared_state->set_on_completed(util::bind(
                            &when_any<Sequence>::on_future_ready, when_.shared_from_this(),
                            idx_, threads::get_self_id()));
                    }
                    else {
                        if (when_.index_.compare_exchange_strong(index, idx_))
                        {
                            when_.goal_reached_on_calling_thread_ = true;
                        }
                    }
                }
                ++idx_;
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

            when_any<Sequence>& when_;
            mutable std::size_t idx_;
        };

        template <typename Sequence>
        void set_on_completed_callback(when_any<Sequence>& when)
        {
            set_when_any_callback_impl<Sequence> callback(when);
            callback.apply(when.lazy_values_.futures);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Sequence>
        struct when_any : boost::enable_shared_from_this<when_any<Sequence> > //-V690
        {
        public:
            void on_future_ready(std::size_t idx, threads::thread_id_type const& id)
            {
                std::size_t index_not_initialized =
                    when_any_result<Sequence>::index_error();
                if (index_.compare_exchange_strong(index_not_initialized, idx))
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
            when_any();
            when_any(when_any const&);

        public:
            typedef Sequence argument_type;
            typedef when_any_result<Sequence> result_type;

            when_any(argument_type && lazy_values)
              : lazy_values_(std::move(lazy_values))
              , index_(when_any_result<Sequence>::index_error())
              , goal_reached_on_calling_thread_(false)
            {}

            result_type operator()()
            {
                // set callback functions to executed when future is ready
                set_on_completed_callback(*this);

                // if one of the requested futures is already set, our
                // callback above has already been called often enough, otherwise
                // we suspend ourselves
                if (!goal_reached_on_calling_thread_)
                {
                    // wait for any of the futures to return to become ready
                    this_thread::suspend(threads::suspended,
                        "hpx::lcos::detail::when_any::operator()");
                }

                // that should not happen
                HPX_ASSERT(index_.load() != when_any_result<Sequence>::index_error());

                lazy_values_.index = index_.load();
                return std::move(lazy_values_);
            }

            result_type lazy_values_;
            boost::atomic<std::size_t> index_;
            bool goal_reached_on_calling_thread_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    lcos::future<when_any_result<std::vector<Future> > >
    when_any(std::vector<Future>& lazy_values, error_code& ec = throws)
    {
        BOOST_STATIC_ASSERT_MSG(
            traits::is_future<Future>::value, "invalid use of when_any");

        typedef std::vector<Future> result_type;

        result_type lazy_values_;
        lazy_values_.reserve(lazy_values.size());
        std::transform(lazy_values.begin(), lazy_values.end(),
            std::back_inserter(lazy_values_),
            detail::when_acquire_future<Future>());

        boost::shared_ptr<detail::when_any<result_type> > f =
            boost::make_shared<detail::when_any<result_type> >(
                std::move(lazy_values_));

        lcos::local::futures_factory<when_any_result<result_type>()> p(
            util::bind(&detail::when_any<result_type>::operator(), f));

        p.apply();
        return p.get_future();
    }

    template <typename Future>
    lcos::future<when_any_result<std::vector<Future> > > //-V659
    when_any(std::vector<Future> && lazy_values, error_code& ec = throws)
    {
        return lcos::when_any(lazy_values, ec);
    }

    template <typename Iterator>
    lcos::future<when_any_result<std::vector<
        typename lcos::detail::future_iterator_traits<Iterator>::type
    > > >
    when_any(Iterator begin, Iterator end, error_code& ec = throws)
    {
        typedef
            typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;
        typedef std::vector<future_type> result_type;

        result_type lazy_values_;
        std::transform(begin, end, std::back_inserter(lazy_values_),
            detail::when_acquire_future<future_type>());

        return lcos::when_any(lazy_values_, ec);
    }

    inline lcos::future<when_any_result<HPX_STD_TUPLE<> > > //-V524
    when_any(error_code& /*ec*/ = throws)
    {
        typedef when_any_result<HPX_STD_TUPLE<> > result_type;

        return lcos::make_ready_future(result_type());
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    lcos::future<when_any_result<std::vector<
        typename lcos::detail::future_iterator_traits<Iterator>::type
    > > >
    when_any_n(Iterator begin, std::size_t count, error_code& ec = throws)
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

        return lcos::when_any(lazy_values_, ec);
    }
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/when_any.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/when_any_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_WAIT_ARGUMENT_LIMIT, <hpx/lcos/when_any.hpp>))                \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

namespace hpx
{
    using lcos::when_any_result;
    using lcos::when_any;
    using lcos::when_any_n;
}

#endif

///////////////////////////////////////////////////////////////////////////////
#else // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

#define HPX_WHEN_ANY_DECAY_FUTURE(Z, N, D)                                    \
    typename util::decay<BOOST_PP_CAT(T, N)>::type                            \
    /**/
#define HPX_WHEN_ANY_ACQUIRE_FUTURE(Z, N, D)                                 \
    detail::when_acquire_future<BOOST_PP_CAT(T, N)>()(BOOST_PP_CAT(f, N))     \
    /**/

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    lcos::future<when_any_result<
        HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WHEN_ANY_DECAY_FUTURE, _)> > >
    when_any(HPX_ENUM_FWD_ARGS(N, T, f), error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<
            BOOST_PP_ENUM(N, HPX_WHEN_ANY_DECAY_FUTURE, _)>
            result_type;

        result_type lazy_values(BOOST_PP_ENUM(N, HPX_WHEN_ANY_ACQUIRE_FUTURE, _));

        boost::shared_ptr<detail::when_any<result_type> > f =
            boost::make_shared<detail::when_any<result_type> >(
                std::move(lazy_values));

        lcos::local::futures_factory<when_any_result<result_type>()> p(
            util::bind(&detail::when_any<result_type>::operator(), f));

        p.apply();
        return p.get_future();
    }
}}

#undef HPX_WHEN_ANY_DECAY_FUTURE
#undef HPX_WHEN_ANY_ACQUIRE_FUTURE
#undef N

#endif

#endif
