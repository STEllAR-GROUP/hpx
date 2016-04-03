//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/when_each.hpp

#if !defined(HPX_LCOS_WHEN_EACH_JUN_16_2014_0206PM)
#define HPX_LCOS_WHEN_EACH_JUN_16_2014_0206PM

#if defined(DOXYGEN)
namespace hpx
{
    /// The function \a when_each is a operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param futures  A vector holding an arbitrary amount of \a future or
    ///                 \a shared_future objects for which \a wait_each should
    ///                 wait.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function.
    ///
    /// \return   Returns a future representing the event of all input futures
    ///           being ready.
    ///
    template <typename F, typename Future>
    future<void> when_each(F&& f, std::vector<Future>&& futures);

    /// The function \a when_each is a operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param begin    The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_each should wait.
    /// \param end      The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_each should wait.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function.
    ///
    /// \return   Returns a future representing the event of all input futures
    ///           being ready.
    ///
    template <typename F, typename Iterator>
    future<Iterator> when_each(F&& f, Iterator begin, Iterator end);

    /// The function \a when_each is a operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param futures  An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a wait_each should wait.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function.
    ///
    /// \return   Returns a future representing the event of all input futures
    ///           being ready.
    ///
    template <typename F, typename... Ts>
    future<void> when_each(F&& f, Ts&&... futures);

    /// The function \a when_each is a operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the event of all those futures
    /// having finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param begin    The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_each_n should wait.
    /// \param count    The number of elements in the sequence starting at
    ///                 \a first.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function.
    ///
    /// \return   Returns a future holding the iterator pointing to the first
    ///           element after the last one.
    ///
    template <typename F, typename Iterator>
    future<Iterator> when_each_n(F&& f, Iterator begin, std::size_t count);
}

#else // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/traits/acquire_shared_state.hpp>
#include <hpx/lcos/when_some.hpp>
#include <hpx/lcos/local/condition_variable.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pack.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/and.hpp>
#include <boost/range/functions.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    namespace detail
    {
        template <typename Tuple, typename F>
        struct when_each_frame //-V690
          : lcos::detail::future_data<void>
        {
            typedef lcos::future<void> type;

        private:
            // workaround gcc regression wrongly instantiating constructors
            when_each_frame();
            when_each_frame(when_each_frame const&);

            template <std::size_t I>
            struct is_end
              : boost::mpl::bool_<
                    util::tuple_size<Tuple>::value == I
                >
            {};

        public:
            template <typename Tuple_, typename F_>
            when_each_frame(Tuple_&& t, F_ && f, std::size_t needed_count)
              : t_(std::forward<Tuple_>(t))
              , f_(std::forward<F_>(f))
              , count_(0)
              , needed_count_(needed_count)
            {}

        protected:
            template <std::size_t I>
            HPX_FORCEINLINE
            void do_await(boost::mpl::true_)
            {
                this->set_value(util::unused);
            }

            // Current element is a range (vector) of futures
            template <std::size_t I, typename Iter>
            void await_range(Iter next, Iter end)
            {
                typedef typename std::iterator_traits<Iter>::value_type
                    future_type;
                typedef typename traits::future_traits<future_type>::type
                    future_result_type;

                void (when_each_frame::*f)(Iter, Iter) =
                    &when_each_frame::await_range<I>;

                for(/**/; next != end; ++next)
                {
                    boost::intrusive_ptr<
                        lcos::detail::future_data<future_result_type>
                    > next_future_data =
                        traits::detail::get_shared_state(*next);

                    if (!next_future_data->is_ready())
                    {
                        next_future_data->execute_deferred();

                        // execute_deferred might have made the future ready
                        if (!next_future_data->is_ready())
                        {
                            // Attach a continuation to this future which will
                            // re-evaluate it and continue to the next argument
                            // (if any).
                            boost::intrusive_ptr<when_each_frame> this_(this);
                            next_future_data->set_on_completed(
                                util::bind(
                                    f, std::move(this_),
                                    std::move(next), std::move(end)));
                            return;
                        }
                    }

                    f_(std::move(*next));
                    ++count_;
                    if(count_ == needed_count_)
                    {
                        do_await<I + 1>(boost::mpl::true_());
                        return;
                    }
                }

                do_await<I + 1>(is_end<I + 1>());
            }

            template <std::size_t I>
            HPX_FORCEINLINE
            void await_next(boost::mpl::false_, boost::mpl::true_)
            {
                await_range<I>(
                    boost::begin(boost::unwrap_ref(util::get<I>(t_))),
                    boost::end(boost::unwrap_ref(util::get<I>(t_))));
            }

            // Current element is a simple future
            template <std::size_t I>
            HPX_FORCEINLINE
            void await_next(boost::mpl::true_, boost::mpl::false_)
            {
                typedef typename util::decay_unwrap<
                    typename util::tuple_element<I, Tuple>::type
                >::type future_type;

                typedef typename traits::future_traits<future_type>::type
                    future_result_type;

                using boost::mpl::false_;
                using boost::mpl::true_;

                future_type& fut = util::get<I>(t_);

                boost::intrusive_ptr<
                    lcos::detail::future_data<future_result_type>
                > next_future_data =
                    traits::detail::get_shared_state(fut);

                if (!next_future_data->is_ready())
                {

                    next_future_data->execute_deferred();

                    // execute_deferred might have made the future ready
                    if (!next_future_data->is_ready())
                    {
                        // Attach a continuation to this future which will
                        // re-evaluate it and continue to the next argument
                        // (if any).
                        void (when_each_frame::*f)(true_, false_) =
                            &when_each_frame::await_next<I>;

                        boost::intrusive_ptr<when_each_frame> this_(this);
                        next_future_data->set_on_completed(
                            hpx::util::bind(
                                f, std::move(this_),
                                true_(), false_()));
                        return;
                    }
                }

                f_(std::move(fut));
                ++count_;
                if(count_ == needed_count_)
                {
                    do_await<I + 1>(boost::mpl::true_());
                    return;
                }

                do_await<I + 1>(is_end<I + 1>());
            }

            template <std::size_t I>
            HPX_FORCEINLINE
            void do_await(boost::mpl::false_)
            {
                typedef typename util::decay_unwrap<
                    typename util::tuple_element<I, Tuple>::type
                >::type future_type;

                typedef typename traits::is_future<future_type>::type is_future;
                typedef typename traits::is_future_range<future_type>::type is_range;

                await_next<I>(is_future(), is_range());
            }

        public:
            HPX_FORCEINLINE void do_await()
            {
                do_await<0>(is_end<0>());
            }

        private:
            Tuple t_;
            F f_;
            std::size_t count_;
            std::size_t needed_count_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Future>
    lcos::future<void>
    when_each(F&& func, std::vector<Future>& lazy_values)
    {
        static_assert(
            traits::is_future<Future>::value, "invalid use of when_each");

        typedef hpx::util::tuple<std::vector<Future> > argument_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each_frame<argument_type, func_type> frame_type;

        std::vector<Future> lazy_values_;
        lazy_values_.reserve(lazy_values.size());

        std::transform(lazy_values.begin(), lazy_values.end(),
            std::back_inserter(lazy_values_),
            traits::acquire_future_disp());

        boost::intrusive_ptr<frame_type> p(new frame_type(
            util::forward_as_tuple(std::move(lazy_values_)),
            std::forward<F>(func), lazy_values_.size()));

        p->do_await();

        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }

    template <typename F, typename Future>
    lcos::future<void> //-V659
    when_each(F&& f, std::vector<Future>&& lazy_values)
    {
        return lcos::when_each(std::forward<F>(f), lazy_values);
    }

    namespace detail
    {
        template <typename Iterator>
        Iterator return_iterator(hpx::future<void>&& fut, Iterator end)
        {
            fut.get();      // rethrow exceptions, if any
            return end;
        }
    }

    template <typename F, typename Iterator>
    lcos::future<Iterator>
    when_each(F&& f, Iterator begin, Iterator end)
    {
        typedef
            typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;

        std::vector<future_type> lazy_values_;
        std::transform(begin, end, std::back_inserter(lazy_values_),
            traits::acquire_future_disp());

        return lcos::when_each(std::forward<F>(f), lazy_values_).then(
            util::bind(&detail::return_iterator<Iterator>,
                util::placeholders::_1, end));
    }

    template <typename F, typename Iterator>
    lcos::future<Iterator>
    when_each_n(F&& f, Iterator begin, std::size_t count)
    {
        typedef
            typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;

        std::vector<future_type> lazy_values_;
        lazy_values_.reserve(count);

        traits::acquire_future_disp func;
        for (std::size_t i = 0; i != count; ++i)
            lazy_values_.push_back(func(*begin++));

        return lcos::when_each(std::forward<F>(f), lazy_values_).then(
            util::bind(&detail::return_iterator<Iterator>,
                util::placeholders::_1, begin));
    }

    template <typename F>
    inline lcos::future<void>
    when_each(F&& f)
    {
        return lcos::make_ready_future();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    typename boost::disable_if<
        boost::mpl::or_<
            traits::is_future<typename util::decay<F>::type>,
            util::detail::any_of<boost::mpl::not_<traits::is_future<Ts> >...>
        >,
        lcos::future<void>
    >::type
    when_each(F&& f, Ts&&... ts)
    {
        typedef hpx::util::tuple<
                typename traits::acquire_future<Ts>::type...
            > argument_type;

        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each_frame<argument_type, func_type> frame_type;

        traits::acquire_future_disp func;
        argument_type lazy_values(func(std::forward<Ts>(ts))...);

        boost::intrusive_ptr<frame_type> p(new frame_type(
            std::move(lazy_values), std::forward<F>(f), sizeof...(Ts)));

        p->do_await();

        using traits::future_access;
        return future_access<typename frame_type::type>::create(std::move(p));
    }
}}

namespace hpx
{
    using lcos::when_each;
    using lcos::when_each_n;
}

#endif // DOXYGEN
#endif

