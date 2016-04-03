//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/when_any.hpp

#if !defined(HPX_LCOS_WHEN_ANY_APR_17_2012_1143AM)
#define HPX_LCOS_WHEN_ANY_APR_17_2012_1143AM

#if defined(DOXYGEN)
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// Result type for \a when_any, contains a sequence of futures and an
    /// index pointing to a ready future.
    template <typename Sequence>
    struct when_any_result
    {
        std::size_t index;  ///< The index of a future which has become ready
        Sequence futures;   ///< The sequence of futures as passed to \a hpx::when_any
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
    ///
    /// \return   Returns a when_any_result holding the same list of futures
    ///           as has been passed to when_any and an index pointing to a
    ///           ready future.
    ///           - future<when_any_result<Container<future<R>>>>: If the input
    ///             cardinality is unknown at compile time and the futures
    ///             are all of the same type. The order of the futures in the
    ///             output container will be the same as given by the input
    ///             iterator.
    template <typename InputIter, typename Container =
        vector<future<typename std::iterator_traits<InputIter>::value_type>>>
    future<when_any_result<Container>>
    when_any(InputIter first, InputIter last);

    /// The function \a when_any is a non-deterministic choice operator. It
    /// OR-composes all future objects given and returns a new future object
    /// representing the same list of futures after one future of that list
    /// finishes execution.
    ///
    /// \param values   [in] A range holding an arbitrary amount of \a futures
    ///                 or \a shared_future objects for which \a when_any should
    ///                 wait.
    ///
    /// \return   Returns a when_any_result holding the same list of futures
    ///           as has been passed to when_any and an index pointing to a
    ///           ready future.
    ///           - future<when_any_result<Container<future<R>>>>: If the input
    ///             cardinality is unknown at compile time and the futures
    ///             are all of the same type. The order of the futures in the
    ///             output container will be the same as given by the input
    ///             iterator.
    template <typename Range>
    future<when_any_result<Range>>
    when_any(Range& values);

    /// The function \a when_any is a non-deterministic choice operator. It
    /// OR-composes all future objects given and returns a new future object
    /// representing the same list of futures after one future of that list
    /// finishes execution.
    ///
    /// \param futures  [in] An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a when_any should wait.
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
    template <typename ...T>
    future<when_any_result<tuple<future<T>...>>>
    when_any(T &&... futures);

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
    ///
    /// \return   Returns a when_any_result holding the same list of futures
    ///           as has been passed to when_any and an index pointing to a
    ///           ready future.
    ///           - future<when_any_result<Container<future<R>>>>: If the input
    ///             cardinality is unknown at compile time and the futures
    ///             are all of the same type. The order of the futures in the
    ///             output container will be the same as given by the input
    ///             iterator.
    ///
    /// \note     None of the futures in the input sequence are invalidated.
    template <typename InputIter, typename Container =
        vector<future<typename std::iterator_traits<InputIter>::value_type>>>
    future<when_any_result<Container>>
    when_any_n(InputIter first, std::size_t count);
}

#else // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/when_any.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/traits/acquire_future.hpp>

#include <boost/atomic.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/utility/swap.hpp>
#include <boost/utility/enable_if.hpp>

#include <algorithm>
#include <cstddef>
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

        when_any_result& operator=(when_any_result const& rhs)
        {
            if (this != &rhs)
            {
                index = rhs.index;
                futures = rhs.futures;
            }
            return true;
        }

        when_any_result& operator=(when_any_result && rhs)
        {
            if (this != &rhs)
            {
                index = rhs.index;
                rhs.index = index_error();
                futures = std::move(rhs.futures);
            }
            return true;
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
            void operator()(Future& future,
                typename boost::enable_if_c<
                    traits::is_future<Future>::value
                >::type* = 0) const
            {
                std::size_t index =
                    when_.index_.load(boost::memory_order_seq_cst);
                if (index == when_any_result<Sequence>::index_error())
                {
                    typedef typename traits::detail::shared_state_ptr_for<
                            Future
                        >::type shared_state_ptr;
                    shared_state_ptr const& shared_state =
                        traits::detail::get_shared_state(future);

                    if (!shared_state->is_ready())
                    {
                        // handle future only if not enough futures are ready
                        // yet also, do not touch any futures which are already
                        // ready

                        shared_state->execute_deferred();

                        // execute_deferred might have made the future ready
                        if (!shared_state->is_ready())
                        {
                            shared_state->set_on_completed(
                                util::bind(
                                    &when_any<Sequence>::on_future_ready,
                                    when_.shared_from_this(),
                                    idx_, threads::get_self_id()));
                            ++idx_;
                            return;
                        }
                    }

                    if (when_.index_.compare_exchange_strong(index, idx_))
                    {
                        when_.goal_reached_on_calling_thread_ = true;
                    }
                }
                ++idx_;
            }

            template <typename Sequence_>
            HPX_FORCEINLINE
            void operator()(Sequence_& sequence,
                typename boost::enable_if_c<
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

            when_any<Sequence>& when_;
            mutable std::size_t idx_;
        };

        template <typename Sequence>
        HPX_FORCEINLINE
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
    template <typename Range>
    typename boost::enable_if<traits::is_future_range<Range>,
        lcos::future<when_any_result<typename util::decay<Range>::type> > >::type
    when_any(Range& lazy_values)
    {
        typedef Range result_type;
        result_type lazy_values_ = traits::acquire_future<result_type>()(lazy_values);

        boost::shared_ptr<detail::when_any<result_type> > f =
            boost::make_shared<detail::when_any<result_type> >(
                std::move(lazy_values_));

        lcos::local::futures_factory<when_any_result<result_type>()> p(
            util::bind(&detail::when_any<result_type>::operator(), f));

        p.apply();
        return p.get_future();
    }

    template <typename Range>
    typename boost::enable_if<traits::is_future_range<Range>,
        lcos::future<when_any_result<typename util::decay<Range>::type> > >::type
    when_any(Range&& lazy_values)
    {
        return lcos::when_any(lazy_values);
    }

    template <typename Iterator, typename Container =
        std::vector<typename lcos::detail::future_iterator_traits<Iterator>::type> >
    lcos::future<when_any_result<Container> >
    when_any(Iterator begin, Iterator end)
    {
        Container lazy_values_;

        typename std::iterator_traits<Iterator>::
            difference_type difference = std::distance(begin, end);
        if (difference > 0)
            traits::detail::reserve_if_vector(
                lazy_values_, static_cast<std::size_t>(difference));

        std::transform(begin, end, std::back_inserter(lazy_values_),
            traits::acquire_future_disp());

        return lcos::when_any(lazy_values_);
    }

    inline lcos::future<when_any_result<hpx::util::tuple<> > > //-V524
    when_any()
    {
        typedef when_any_result<hpx::util::tuple<> > result_type;

        return lcos::make_ready_future(result_type());
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator, typename Container =
        std::vector<typename lcos::detail::future_iterator_traits<Iterator>::type> >
    lcos::future<when_any_result<Container> >
    when_any_n(Iterator begin, std::size_t count)
    {
        Container lazy_values_;
        traits::detail::reserve_if_vector(lazy_values_, count);

        traits::acquire_future_disp func;
        for (std::size_t i = 0; i != count; ++i)
            lazy_values_.push_back(func(*begin++));

        return lcos::when_any(lazy_values_);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    lcos::future<when_any_result<
        hpx::util::tuple<typename traits::acquire_future<Ts>::type...>
    > >
    when_any(Ts&&... ts)
    {
        typedef hpx::util::tuple<
                typename traits::acquire_future<Ts>::type...
            > result_type;

        traits::acquire_future_disp func;
        result_type lazy_values(func(std::forward<Ts>(ts))...);

        boost::shared_ptr<detail::when_any<result_type> > f =
            boost::make_shared<detail::when_any<result_type> >(
                std::move(lazy_values));

        lcos::local::futures_factory<when_any_result<result_type>()> p(
            util::bind(&detail::when_any<result_type>::operator(), f));

        p.apply();
        return p.get_future();
    }
}}

namespace hpx
{
    using lcos::when_any_result;
    using lcos::when_any;
    using lcos::when_any_n;
}

#endif // DOXYGEN
#endif
