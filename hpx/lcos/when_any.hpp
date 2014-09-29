//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/when_any.hpp

#if defined(DOXYGEN)
namespace hpx
{
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
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to when_any.
    ///           - future<vector<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type. The order of the futures in the output vector
    ///             will be the same as given by the input iterator.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    template <typename InputIter>
    future<vector<future<typename std::iterator_traits<InputIter>::value_type>>>
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
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to when_any.
    ///           - future<vector<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    template <typename R>
    future<std::vector<future<R>> >
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
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to when_any.
    ///           - future<tuple<future<T0>, future<T1>, future<T2>...>>: If
    ///             inputs are fixed in number and are of heterogeneous types.
    ///             The inputs can be any arbitrary number of future objects.
    ///           - future<tuple<>> if \a when_any is called with zero arguments.
    ///             The returned future will be initially ready.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    template <typename ...T>
    future<tuple<future<T>...>>
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
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to when_any_n.
    ///           - future<vector<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type. The order of the futures in the output vector
    ///             will be the same as given by the input iterator.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    ///
    /// \note     None of the futures in the input sequence are invalidated.
    template <typename InputIter>
    future<vector<future<typename std::iterator_traits<InputIter>::value_type>>>
    when_any_n(InputIter first, std::size_t count, error_code& ec = throws);

    /// The function \a when_any_back is a non-deterministic choice
    /// operator. It OR-composes all future objects given and returns the same
    /// list of futures after one future of that list finishes execution. The
    /// future object that was first detected as being ready swaps its
    /// position with that of the last element of the result collection, so
    /// that the ready future object may be identified in constant time.
    ///
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_any_back should wait.
    /// \param last     [in] The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_any_back should wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to when_any_back.
    ///           - future<vector<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type. The order of the futures in the output vector
    ///             will be the same as given by the input iterator.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    template <typename InputIter>
    future<vector<future<typename std::iterator_traits<InputIter>::value_type>>>
    when_any_back(InputIter first, InputIter last, error_code& ec = throws);

    /// The function \a when_any_back is a non-deterministic choice
    /// operator. It OR-composes all future objects given and returns the same
    /// list of futures after one future of that list finishes execution. The
    /// future object that was first detected as being ready swaps its
    /// position with that of the last element of the result collection, so
    /// that the ready future object may be identified in constant time.
    ///
    /// \param futures  [in] A vector holding an arbitrary amount of \a future or
    ///                 \a shared_future objects for which \a when_any_back should
    ///                 wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to when_any_back.
    ///           - future<vector<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    template <typename R>
    future<std::vector<future<R>> >
    when_any_back(std::vector<future<R>>& futures, error_code& ec = throws);

    /// The function \a when_any_back is a non-deterministic choice
    /// operator. It OR-composes all future objects given and returns the same
    /// list of futures after one future of that list finishes execution. The
    /// future object that was first detected as being ready swaps its
    /// position with that of the last element of the result collection, so
    /// that the ready future object may be identified in constant time.
    ///
    /// \param futures  [in] An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a when_any should wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to when_any.
    ///           - future<tuple<future<T0>, future<T1>, future<T2>...>>: If
    ///             inputs are fixed in number and are of heterogeneous types.
    ///             The inputs can be any arbitrary number of future objects.
    ///           - future<tuple<>> if \a when_any is called with zero arguments.
    ///             The returned future will be initially ready.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    template <typename ...T>
    future<tuple<future<T>...>>
    when_any_back(T &&... futures, error_code& ec = throws);

    /// The function \a when_any_back_n is a non-deterministic choice
    /// operator. It OR-composes all future objects given and returns the same
    /// list of futures after one future of that list finishes execution. The
    /// future object that was first detected as being ready swaps its
    /// position with that of the last element of the result collection, so
    /// that the ready future object may be identified in constant time.
    ///
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a when_any_back_n should wait.
    /// \param count    [in] The number of elements in the sequence starting at
    ///                 \a first.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \note The function \a when_any_back_n returns after at least one future has
    ///       become ready. All input futures are still valid after \a when_any_back_n
    ///       returns.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to when_any_back_n.
    ///           - future<vector<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type. The order of the futures in the output vector
    ///             will be the same as given by the input iterator.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    ///
    /// \note     None of the futures in the input sequence are invalidated.
    template <typename InputIter>
    future<vector<future<typename std::iterator_traits<InputIter>::value_type>>>
    when_any_back_n(InputIter first, std::size_t count, error_code& ec = throws);
}
#else

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_WHEN_ANY_APR_17_2012_1143AM)
#define HPX_LCOS_WHEN_ANY_APR_17_2012_1143AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/when_some.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

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
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Future>
        struct when_any_back //-V690
          : boost::enable_shared_from_this<when_any_back<Future> >
        {
        private:
            enum { index_error = -1 };

            void on_future_ready(std::size_t idx, threads::thread_id_type const& id)
            {
                std::size_t index_not_initialized =
                    static_cast<std::size_t>(index_error);
                if (index_.compare_exchange_strong(index_not_initialized, idx))
                {
                    // reactivate waiting thread only if it's not us
                    if (id != threads::get_self_id())
                        threads::set_thread_state(id, threads::pending);
                }
            }

        private:
            // workaround gcc regression wrongly instantiating constructors
            when_any_back();
            when_any_back(when_any_back const&);

        public:
            typedef std::vector<Future> result_type;
            typedef std::vector<Future> argument_type;

            when_any_back(argument_type && lazy_values)
              : lazy_values_(std::move(lazy_values))
              , index_(static_cast<std::size_t>(index_error))
            {}

            result_type operator()()
            {
                index_.store(static_cast<std::size_t>(index_error));

                std::size_t size = lazy_values_.size();

                // set callback functions to execute when future is ready
                threads::thread_id_type id = threads::get_self_id();
                for (std::size_t i = 0; i != size; ++i)
                {
                    if (lazy_values_[i].is_ready())
                    {
                        index_.store(i);
                        break;
                    } else {
                        typedef
                            typename lcos::detail::shared_state_ptr_for<Future>::type
                            shared_state_ptr;

                        shared_state_ptr const& shared_state =
                            lcos::detail::get_shared_state(lazy_values_[i]);

                        shared_state->set_on_completed(util::bind(
                            &when_any_back::on_future_ready,
                            this->shared_from_this(), i, id));
                    }
                }

                // If one of the futures is already set, our callback above has
                // already been called, otherwise we suspend ourselves
                if (index_.load() == static_cast<std::size_t>(index_error))
                {
                    // wait for any of the futures to return to become ready
                    this_thread::suspend(threads::suspended);
                }

                // that should not happen
                HPX_ASSERT(index_.load() != static_cast<std::size_t>(index_error));

                boost::swap(lazy_values_[index_], lazy_values_.back());
                return std::move(lazy_values_);
            }

            std::vector<Future> lazy_values_;
            boost::atomic<std::size_t> index_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    lcos::future<std::vector<Future> >
    when_any(std::vector<Future>& lazy_values, error_code& ec = throws)
    {
        typedef std::vector<Future> result_type;

        if (lazy_values.empty())
            return lcos::make_ready_future(result_type());

        return lcos::when_some(1, lazy_values, ec);
    }

    template <typename Future>
    lcos::future<std::vector<Future> > //-V659
    when_any(std::vector<Future> && lazy_values, error_code& ec = throws)
    {
        return lcos::when_any(lazy_values, ec);
    }

    template <typename Iterator>
    lcos::future<std::vector<
        typename lcos::detail::future_iterator_traits<Iterator>::type
    > >
    when_any(Iterator begin, Iterator end, error_code& ec = throws)
    {
        return lcos::when_some(1, begin, end, ec);
    }

    inline lcos::future<HPX_STD_TUPLE<> > //-V524
    when_any(error_code& /*ec*/ = throws)
    {
        typedef HPX_STD_TUPLE<> result_type;

        return lcos::make_ready_future(result_type());
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    lcos::future<std::vector<
        typename lcos::detail::future_iterator_traits<Iterator>::type
    > >
    when_any_n(Iterator begin, std::size_t count, error_code& ec = throws)
    {
        return when_some_n(1, begin, count, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    lcos::future<std::vector<Future> >
    when_any_back(std::vector<Future>& lazy_values, error_code& ec = throws)
    {
        typedef std::vector<Future> result_type;

        if (lazy_values.empty())
            return lcos::make_ready_future(std::move(lazy_values));

        result_type lazy_values_;
        std::transform(lazy_values.begin(), lazy_values.end(),
            std::back_inserter(lazy_values_),
            detail::when_acquire_future<Future>());

        boost::shared_ptr<detail::when_any_back<Future> > f =
            boost::make_shared<detail::when_any_back<Future> >(
                std::move(lazy_values_));

        lcos::local::futures_factory<result_type()> p(
            util::bind(&detail::when_any_back<Future>::operator(), f));

        p.apply();
        return p.get_future();
    }

    template <typename Future>
    lcos::future<std::vector<Future> > //-V659
    when_any_back(std::vector<Future> && lazy_values, error_code& ec = throws)
    {
        return lcos::when_any_back(lazy_values, ec);
    }

    template <typename Iterator>
    lcos::future<std::vector<
        typename lcos::detail::future_iterator_traits<Iterator>::type
    > >
    when_any_back(Iterator begin, Iterator end, error_code& ec = throws)
    {
        typedef
            typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;
        typedef std::vector<future_type> result_type;

        result_type lazy_values_;
        std::transform(begin, end, std::back_inserter(lazy_values_),
            detail::when_acquire_future<future_type>());
        return lcos::when_any_back(lazy_values_, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    lcos::future<std::vector<
        typename lcos::detail::future_iterator_traits<Iterator>::type
    > >
    when_any_back_n(Iterator begin, std::size_t count, error_code& ec = throws)
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

        return lcos::when_any_back(lazy_values_, ec);
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
    using lcos::when_any;
    using lcos::when_any_back;
    using lcos::when_any_n;
    using lcos::when_any_back_n;
}

#endif

///////////////////////////////////////////////////////////////////////////////
#else // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

#define HPX_WHEN_SOME_DECAY_FUTURE(Z, N, D)                                   \
    typename util::decay<BOOST_PP_CAT(T, N)>::type                            \
    /**/

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    lcos::future<HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WHEN_SOME_DECAY_FUTURE, _)> >
    when_any(HPX_ENUM_FWD_ARGS(N, T, f), error_code& ec = throws)
    {
        return lcos::when_some(1, HPX_ENUM_FORWARD_ARGS(N, T, f), ec);
    }
}}

#undef HPX_WHEN_SOME_DECAY_FUTURE
#undef N

#endif

#endif
