//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_WHEN_EACH_JUN_16_2014_0206PM)
#define HPX_LCOS_WHEN_EACH_JUN_16_2014_0206PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/when_some.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Sequence, typename F>
        struct when_each;

        template <typename Sequence, typename F>
        struct set_when_each_on_completed_callback_impl
        {
            explicit set_when_each_on_completed_callback_impl(
                    boost::shared_ptr<when_each<Sequence, F> > when)
              : when_(when)
            {}

            template <typename Future>
            static void on_future_ready(Future& future,
                boost::shared_ptr<when_each<Sequence, F> > when,
                threads::thread_id_type const& id)
            {
                when->f_(std::move(future));

                if (when->count_.fetch_add(1) + 1 == when->needed_count_)
                {
                    // reactivate waiting thread only if it's not us
                    if (id != threads::get_self_id())
                        threads::set_thread_state(id, threads::pending);
                    else
                        when->goal_reached_on_calling_thread_ = true;
                }
            }

            template <typename Future>
            void operator()(Future& future) const
            {
                std::size_t counter = when_->count_.load(boost::memory_order_seq_cst);
                if (counter < when_->needed_count_ && !future.is_ready()) {
                    // handle future only if not enough futures are ready yet
                    // also, do not touch any futures which are already ready

                    typedef
                        typename lcos::detail::shared_state_ptr_for<Future>::type
                        shared_state_ptr;

                    shared_state_ptr const& shared_state =
                        lcos::detail::get_shared_state(future);

                    shared_state->set_on_completed(util::bind(
                        &set_when_each_on_completed_callback_impl::on_future_ready<Future>,
                        boost::ref(future), when_, threads::get_self_id()));
                }
                else {
                    if (when_->count_.fetch_add(1) + 1 == when_->needed_count_)
                    {
                        when_->goal_reached_on_calling_thread_ = true;
                    }
                    when_->f_(std::move(future));    // invoke callback right away
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

            boost::shared_ptr<when_each<Sequence, F> > when_;
        };

        template <typename Sequence, typename F>
        void set_on_completed_callback(
            boost::shared_ptr<when_each<Sequence, F> > when)
        {
            set_when_each_on_completed_callback_impl<Sequence, F>
                set_on_completed_callback_helper(when);
            set_on_completed_callback_helper.apply(when->lazy_values_);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Sequence, typename F>
        struct when_each
          : boost::enable_shared_from_this<when_each<Sequence, F> >
        {
        private:
            HPX_MOVABLE_BUT_NOT_COPYABLE(when_each)

        public:
            typedef Sequence argument_type;
            typedef void result_type;

            template <typename F_>
            when_each(argument_type && lazy_values, F_ && f, std::size_t n)
              : lazy_values_(std::move(lazy_values)),
                count_(0),
                f_(std::forward<F>(f)),
                needed_count_(n),
                goal_reached_on_calling_thread_(false)
            {}

            result_type operator()()
            {
                count_.store(0);
                goal_reached_on_calling_thread_ = false;

                // set callback functions to executed when future is ready
                set_on_completed_callback(this->shared_from_this());

                // If all of the requested futures are already set then our
                // callback above has already been called, otherwise we suspend
                // ourselves.
                if (!goal_reached_on_calling_thread_)
                {
                    // wait for all of the futures to return to become ready
                    this_thread::suspend(threads::suspended,
                        "hpx::lcos::detail::when_each::operator()");
                }

                // all futures should be ready
                HPX_ASSERT(count_.load(boost::memory_order_seq_cst) == needed_count_);
            }

            argument_type lazy_values_;
            boost::atomic<std::size_t> count_;
            F f_;
            std::size_t const needed_count_;
            bool goal_reached_on_calling_thread_;
        };
    }

    /// The function \a when_each is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns a new future object representing the same list of futures
    /// after n of them finished executing. It also calls the supplied callback
    /// for each of the futures which becomes ready.
    ///
    /// \note There are four variations of when_each. The first takes a pair
    ///       of InputIterators. The second takes an std::vector of future<R>.
    ///       The third takes any arbitrary number of future<R>, where R need
    ///       not be the same type. The fourth takes an iterator and a count.
    ///
    /// \return   Returns a future holding either nothing or the iterator
    ///           pointing to the first element after the last one.
    ///
    template <typename Future, typename F>
    lcos::future<void>
    when_each(std::vector<Future>& lazy_values, F && func)
    {
        BOOST_STATIC_ASSERT_MSG(
            traits::is_future<Future>::value, "invalid use of when_each");

        typedef void result_type;
        typedef std::vector<Future> argument_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;

        std::vector<Future> lazy_values_;
        lazy_values_.reserve(lazy_values.size());

        std::transform(lazy_values.begin(), lazy_values.end(),
            std::back_inserter(lazy_values_),
            detail::when_acquire_future<Future>());

        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values_),
                std::forward<F>(func), lazy_values_.size());

        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));

        p.apply();
        return p.get_future();
    }

    template <typename Future, typename F>
    lcos::future<void> //-V659
    when_each(std::vector<Future> && lazy_values, F && f)
    {
        return lcos::when_each(lazy_values, std::forward<F>(f));
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

    template <typename Iterator, typename F>
    lcos::future<Iterator>
    when_each(Iterator begin, Iterator end, F && f)
    {
        typedef
            typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;

        std::vector<future_type> lazy_values_;
        std::transform(begin, end, std::back_inserter(lazy_values_),
            detail::when_acquire_future<future_type>());

        return lcos::when_each(lazy_values_, std::forward<F>(f)).then(
            util::bind(&detail::return_iterator<Iterator>,
                util::placeholders::_1, end));
    }

    template <typename Iterator, typename F>
    lcos::future<Iterator>
    when_each_n(Iterator begin, std::size_t count, F && f)
    {
        typedef
            typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;

        std::vector<future_type> lazy_values_;
        lazy_values_.reserve(count);
        detail::when_acquire_future<future_type> func;
        for (std::size_t i = 0; i != count; ++i)
            lazy_values_.push_back(func(*begin++));

        return lcos::when_each(lazy_values_, std::forward<F>(f)).then(
            util::bind(&detail::return_iterator<Iterator>,
                util::placeholders::_1, begin));
    }

    template <typename F>
    inline lcos::future<void>
    when_each(F && f)
    {
        return lcos::make_ready_future();
    }
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/when_each.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/when_each_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_WAIT_ARGUMENT_LIMIT, <hpx/lcos/when_each.hpp>))               \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

namespace hpx
{
    using lcos::when_each;
    using lcos::when_each_n;
}

#endif

///////////////////////////////////////////////////////////////////////////////
#else // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

#define HPX_WHEN_EACH_DECAY_FUTURE(Z, N, D)                                   \
    typename util::decay<BOOST_PP_CAT(T, N)>::type                            \
    /**/
#define HPX_WHEN_EACH_ACQUIRE_FUTURE(Z, N, D)                                 \
    detail::when_acquire_future<BOOST_PP_CAT(T, N)>()(BOOST_PP_CAT(f, N))     \
    /**/

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename T), typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >,
        lcos::future<void>
    >::type
    when_each(HPX_ENUM_FWD_ARGS(N, T, f), F && func)
    {
        typedef HPX_STD_TUPLE<
            BOOST_PP_ENUM(N, HPX_WHEN_EACH_DECAY_FUTURE, _)>
            argument_type;
        typedef void result_type;
        typedef typename util::decay<F>::type func_type;
        typedef detail::when_each<argument_type, func_type> when_each_type;

        argument_type lazy_values(BOOST_PP_ENUM(N, HPX_WHEN_EACH_ACQUIRE_FUTURE, _));

        boost::shared_ptr<when_each_type> f =
            boost::make_shared<when_each_type>(std::move(lazy_values),
                std::forward<F>(func), N);

        lcos::local::futures_factory<result_type()> p(
            util::bind(&when_each_type::operator(), f));

        p.apply();
        return p.get_future();
    }
}}

#undef HPX_WHEN_EACH_DECAY_FUTURE
#undef HPX_WHEN_EACH_ACQUIRE_FUTURE
#undef N

#endif

