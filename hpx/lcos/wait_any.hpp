//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_when_any_APR_17_2012_1143AM)
#define HPX_LCOS_when_any_APR_17_2012_1143AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <vector>

#include <boost/assert.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/iterate.hpp>

#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/accumulate.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/fusion/include/tuple.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename RT>
        struct when_any
        {
        private:
            BOOST_MOVABLE_BUT_NOT_COPYABLE(when_any)

            enum { index_error = std::size_t(-1) };

            void on_future_ready(std::size_t idx, threads::thread_id_type id)
            {
                mutex_type::scoped_lock l(mtx_);
                if (index_ == index_error) {
                    index_ = idx;

                    // reactivate waiting thread only if it's not us
                    if (id != threads::get_self().get_thread_id())
                        threads::set_thread_state(id, threads::pending);
                }
            }

        public:
            typedef lcos::local::spinlock mutex_type;
            typedef HPX_STD_TUPLE<int, lcos::future<T, RT> > result_type;
            typedef std::vector<lcos::future<T, RT> > argument_type;

            when_any(argument_type const& lazy_values)
              : lazy_values_(lazy_values), index_(index_error)
            {}

            when_any(BOOST_RV_REF(argument_type) lazy_values)
              : lazy_values_(boost::move(lazy_values)), index_(index_error)
            {}

            when_any(BOOST_RV_REF(when_any) rhs)
              : lazy_values_(boost::move(rhs.lazy_values_)),
                index_(rhs.index_),
                mtx_(boost::move(rhs.mtx_))
            {
                rhs.index_ = index_error;
            }

            when_any& operator= (BOOST_RV_REF(when_any) rhs)
            {
                if (this != &rhs) {
                    mutex_type::scoped_lock l1(mtx_);
                    mutex_type::scoped_lock l2(rhs.mtx_);
                    lazy_values_ = boost::move(rhs.lazy_values_);
                    index_ = rhs.index_;
                    rhs.index_ = index_error;
                    mtx_ = boost::move(rhs.mtx_);
                }
                return *this;
            }

            result_type operator()()
            {
                mutex_type::scoped_lock l(mtx_);
                index_ = index_error;

                {
                    util::unlock_the_lock<mutex_type::scoped_lock> ul(l);

                    // set callback functions to execute when future is ready
                    threads::thread_id_type id = threads::get_self().get_thread_id();
                    for (std::size_t i = 0; i < lazy_values_.size(); ++i)
                    {
                        lazy_values_[i].when(
                            util::bind(&when_any::on_future_ready, this, i, id)
                        );
                    }
                }

                // if one of the futures is already set, our callback above has
                // already been called, otherwise we suspend ourselves
                if (index_ == index_error)
                {
                    // wait for any of the futures to return to become ready
                    util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
                    this_thread::suspend(threads::suspended);
                }

                BOOST_ASSERT(index_ != index_error);       // that should not happen

                // reset all pending callback functions
                l.unlock();
                for (std::size_t i = 0; i < lazy_values_.size(); ++i)
                    lazy_values_[i].when();

                return result_type(static_cast<int>(index_), lazy_values_[index_]);
            }

            std::vector<lcos::future<T, RT> > lazy_values_;
            std::size_t index_;
            mutable mutex_type mtx_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Tuple, typename T, typename RT>
        struct when_any_tuple
        {
        private:
            BOOST_MOVABLE_BUT_NOT_COPYABLE(when_any_tuple)

            struct init_when
            {
                init_when(when_any_tuple& outer, threads::thread_id_type id)
                  : outer_(&outer), id_(id)
                {}

                typedef std::size_t result_type;

                result_type operator()(std::size_t i, lcos::future<T, RT> f) const
                {
                    f.when(util::bind(&when_any_tuple::on_future_ready,
                        outer_, ++i, id_));
                    return i;
                };

                when_any_tuple* outer_;
                threads::thread_id_type id_;
            };
            friend struct init_when;

            struct reset_when
            {
                typedef void result_type;

                result_type operator()(lcos::future<T, RT> f) const
                {
                    f.when();
                };
            };

            enum { index_error = std::size_t(-1) };

            void on_future_ready(std::size_t idx, threads::thread_id_type id)
            {
                mutex_type::scoped_lock l(mtx_);
                if (index_ == index_error) {
                    index_ = idx;

                    // reactivate waiting thread only if it's not us
                    if (id != threads::get_self().get_thread_id())
                        threads::set_thread_state(id, threads::pending);
                }
            }

            // extract element idx from given tuple
            template <typename First, typename Last>
            static lcos::future<T, RT> const&
            get_element(First const& first, Last const& last, std::size_t idx,
                boost::mpl::false_)
            {
                if (idx == 0)
                    return boost::fusion::deref(first);

                return get_element(boost::fusion::next(first), last, --idx,
                    boost::fusion::result_of::equal_to<
                        typename boost::fusion::result_of::next<First>::type,
                        Last
                    >());
            }

            template <typename First, typename Last>
            static lcos::future<T, RT> const&
            get_element(First const&, Last const&, std::size_t,
                boost::mpl::true_)
            {
                static lcos::future<T, RT> f;
                return f;   // shouldn't ever be called
            }

            static lcos::future<T, RT> const&
            get_element(Tuple const& t, std::size_t idx)
            {
                BOOST_ASSERT(idx < boost::fusion::result_of::size<Tuple>::value);
                return get_element(
                    boost::fusion::begin(t), boost::fusion::end(t), idx,
                    boost::fusion::result_of::equal_to<
                        typename boost::fusion::result_of::begin<Tuple>::type,
                        typename boost::fusion::result_of::end<Tuple>::type>());
            }

        public:
            typedef lcos::local::spinlock mutex_type;
            typedef HPX_STD_TUPLE<int, lcos::future<T, RT> > result_type;
            typedef Tuple argument_type;

            when_any_tuple(argument_type const& lazy_values)
              : lazy_values_(lazy_values), index_(index_error)
            {}

            when_any_tuple(BOOST_RV_REF(argument_type) lazy_values)
              : lazy_values_(boost::move(lazy_values)), index_(index_error)
            {}

            when_any_tuple(BOOST_RV_REF(when_any_tuple) rhs)
              : lazy_values_(boost::move(rhs.lazy_values_)),
                index_(rhs.index_),
                mtx_(boost::move(rhs.mtx_))
            {
                rhs.index_ = index_error;
            }

            when_any_tuple& operator= (BOOST_RV_REF(when_any_tuple) rhs)
            {
                if (this != &rhs) {
                    mutex_type::scoped_lock l1(mtx_);
                    mutex_type::scoped_lock l2(rhs.mtx_);
                    lazy_values_ = boost::move(rhs.lazy_values_);
                    index_ = rhs.index_;
                    rhs.index_ = index_error;
                    mtx_ = boost::move(rhs.mtx_);
                }
                return *this;
            }

        public:
            result_type operator()()
            {
                mutex_type::scoped_lock l(mtx_);
                index_ = index_error;

                {
                    util::unlock_the_lock<mutex_type::scoped_lock> ul(l);

                    // set callback functions to execute when future is ready
                    std::size_t tmp = index_error;
                    boost::fusion::accumulate(lazy_values_, tmp,
                        init_when(*this, threads::get_self_id()));
                }

                // if one of the futures is already set, our callback above has
                // already been called, otherwise we suspend ourselves
                if (index_ == index_error)
                {
                    // wait for any of the futures to return to become ready
                    util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
                    this_thread::suspend(threads::suspended);
                }

                BOOST_ASSERT(index_ != index_error);       // that should not happen

                // reset all pending callback functions
                l.unlock();
                boost::fusion::for_each(lazy_values_, reset_when());

                return result_type(static_cast<int>(index_),
                    get_element(lazy_values_, index_));
            }

            Tuple lazy_values_;
            std::size_t index_;
            mutable mutex_type mtx_;
        };
    }

    /// The function \a when_any is a non-deterministic choice operator. It
    /// OR-composes all future objects stored in the given vector and returns
    /// a new future object representing the first future from that list which
    /// finishes execution.
    ///
    /// \return   The returned future holds a pair of values, the first value
    ///           is the index of the future which returned first and the second
    ///           value represents the actual future which returned first.
    template <typename T, typename RT>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T, RT> > >
    when_any(BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<T, RT> >))) lazy_values)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T, RT> > return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_any<T, RT>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }

    template <typename T, typename RT>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T, RT> > >
    when_any(std::vector<lcos::future<T, RT> > const& lazy_values)
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T, RT> > return_type;
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                detail::when_any<T, RT>(lazy_values));
        p.apply();
        return p.get_future();
    }

    /// The function \a wait_any is a non-deterministic choice operator. It
    /// OR-composes all future objects stored in the given vector and returns
    /// a new future object representing the first future from that list which
    /// finishes execution.
    ///
    /// \return   The returned tuple holds a pair of values, the first value
    ///           is the index of the future which returned first and the second
    ///           value represents the actual future which returned first.
    template <typename T, typename RT>
    HPX_STD_TUPLE<int, lcos::future<T, RT> >
    wait_any(BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<T, RT> >))) lazy_values)
    {
        return when_any(lazy_values).get();
    }

    template <typename T, typename RT>
    HPX_STD_TUPLE<int, lcos::future<T, RT> >
    wait_any(std::vector<lcos::future<T, RT> > const& lazy_values)
    {
        return when_any(lazy_values).get();
    }
}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/wait_any.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/wait_any_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_WAIT_ARGUMENT_LIMIT, <hpx/lcos/wait_any.hpp>))                \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

///////////////////////////////////////////////////////////////////////////////
#else // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

#define HPX_WHEN_ANY_FUTURE_ARG(z, n, _)                                      \
        lcos::future<T, RT> BOOST_PP_CAT(f, n)                                \
    /**/
#define HPX_WHEN_ANY_FUTURE_TYPE(z, n, _)                                     \
        lcos::future<T, RT>                                                   \
    /**/
#define HPX_WHEN_ANY_FUTURE_VAR(z, n, _) BOOST_PP_CAT(f, n)                   \
    /**/

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename RT>
    lcos::future<HPX_STD_TUPLE<int, lcos::future<T, RT> > >
    when_any (BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_ARG, _))
    {
        typedef HPX_STD_TUPLE<int, lcos::future<T, RT> > return_type;
        typedef boost::fusion::tuple<
            BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_TYPE, _)
        > argument_type;

        lcos::local::futures_factory<return_type()> p(
            detail::when_any_tuple<argument_type, T, RT>(
                argument_type(BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_VAR, _))));
        p.apply();
        return p.get_future();
    }

    template <typename T, typename RT>
    HPX_STD_TUPLE<int, lcos::future<T, RT> >
    wait_any (BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_ARG, _))
    {
        return when_any(BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_VAR, _)).get();
    }
}

#undef HPX_WHEN_ANY_FUTURE_ARG
#undef HPX_WHEN_ANY_FUTURE_TYPE
#undef HPX_WHEN_ANY_FUTURE_VAR
#undef N

#endif

