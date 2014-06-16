//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_UTIL_LOOP_MAY_27_2014_1040PM)
#define HPX_STL_UTIL_LOOP_MAY_27_2014_1040PM

#include <hpx/hpx_fwd.hpp>

#include <iterator>
#include <algorithm>

#include <boost/atomic.hpp>

namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // cancellation_token is used for premature cancellation of algorithms
    class cancellation_token
    {
    private:
        boost::atomic<bool> was_cancelled_;

    public:
        cancellation_token()
          : was_cancelled_(false)
        {}

        bool was_cancelled() const BOOST_NOEXCEPT
        {
            return was_cancelled_.load(boost::memory_order_relaxed);
        }

        void cancel() BOOST_NOEXCEPT
        {
            was_cancelled_.store(true, boost::memory_order_relaxed);
        }
    };

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function starting from a given
        // iterator position.
        template <typename IterCat>
        struct loop
        {
            ///////////////////////////////////////////////////////////////////
            template <typename Iter, typename F>
            static Iter call(Iter it, Iter end, F && f, boost::mpl::false_)
            {
                for (/**/; it != end; ++it)
                    f(*it);

                return it;
            }

            template <typename Iter, typename F>
            static Iter call(Iter it, Iter end, F && f, boost::mpl::true_)
            {
                typedef typename std::iterator_traits<Iter>::value_type type;
                hpx::wait_each(it, end,
                    [&f](hpx::future<type> fut)
                    {
                        f(fut.get());
                        return true;
                    });
                return end;
            }

            ///////////////////////////////////////////////////////////////////////
            template <typename Iter, typename F, typename CancelToken>
            static Iter call(Iter it, Iter end, F && func, CancelToken& tok)
            {
                for (/**/; it != end; ++it)
                {
                    func(*it);
                    if (tok.was_cancelled())
                        break;
                }
                return it;
            }
        };

        ///////////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position.
        template <typename IterCat>
        struct loop_n
        {
            ///////////////////////////////////////////////////////////////////
            // handle sequences of non-futures
            template <typename Iter, typename F>
            static Iter call(Iter it, std::size_t count, F && f,
                boost::mpl::false_)
            {
                for (/**/; count != 0; --count, ++it)
                    f(*it);

                return it;
            }

            template <typename Iter, typename F, typename CancelToken>
            static Iter call(Iter it, std::size_t count, F && f,
                CancelToken& tok, boost::mpl::false_)
            {
                for (/**/; count != 0; --count, ++it)
                {
                    f(*it);
                    if (tok.was_cancelled())
                        break;
                }
                return it;
            }

            ///////////////////////////////////////////////////////////////////
            // handle sequences of futures
            template <typename Iter, typename F>
            static Iter call(Iter it, std::size_t count, F && f,
                boost::mpl::true_)
            {
                typedef typename std::iterator_traits<Iter>::value_type type;
                return hpx::wait_each_n(it, count,
                    [&f](hpx::future<type> fut)
                    {
                        f(fut.get());
                        return true;
                    });
            }

            template <typename Iter, typename F, typename CancelToken>
            static Iter call(Iter it, std::size_t count, F && f,
                CancelToken& tok, boost::mpl::true_)
            {
                typedef typename std::iterator_traits<Iter>::value_type type;
                return hpx::wait_each_n(it, count,
                    [&f, &tok](hpx::future<type> fut)
                    {
                        f(fut.get());
                        return !tok.was_cancelled();
                    });
            }
        };

        // specialization for random access iterators
        template <>
        struct loop_n<std::random_access_iterator_tag>
        {
            ///////////////////////////////////////////////////////////////////
            // handle sequences of non-futures
            template <typename Iter, typename F>
            static Iter call(Iter it, std::size_t count, F && f,
                boost::mpl::false_)
            {
                for (std::size_t i = 0; i != count; ++i)
                    f(it[i]);

                std::advance(it, count);
                return it;
            }

            template <typename Iter, typename F, typename CancelToken>
            static Iter call(Iter it, std::size_t count, F && func,
                CancelToken& tok, boost::mpl::false_)
            {
                std::size_t i = 0;
                for (/**/; i != count; ++i)
                {
                    func(it[i]);
                    if (tok.was_cancelled())
                        break;
                }
                std::advance(it, i);
                return it;
            }

            ///////////////////////////////////////////////////////////////////
            // handle sequence of futures
            template <typename Iter, typename F>
            static Iter call(Iter it, std::size_t count, F && f,
                boost::mpl::true_)
            {
                typedef typename std::iterator_traits<Iter>::value_type type;
                return hpx::wait_each_n(it, count,
                    [&f](hpx::future<type> fut)
                    {
                        f(fut.get());
                        return true;
                    });
            }

            template <typename Iter, typename F, typename CancelToken>
            static Iter call(Iter it, std::size_t count, F && func,
                CancelToken& tok, boost::mpl::true_)
            {
                typedef typename std::iterator_traits<Iter>::value_type type;
                return hpx::wait_each_n(it, count,
                    [&f, &tok](hpx::future<type> fut)
                    {
                        f(fut.get());
                        return !tok.was_cancelled();
                    });
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename F>
    BOOST_FORCEINLINE Iter
    loop_n(Iter it, std::size_t count, F && f)
    {
        typedef typename std::iterator_traits<Iter>::iterator_category cat;
        typedef typename hpx::traits::is_future<
            typename std::iterator_traits<Iter>::value_type
        >::type pred;

        return detail::loop_n<cat>::call(it, count, std::forward<F>(f), pred());
    }

    template <typename Iter, typename F, typename CancelToken>
    BOOST_FORCEINLINE Iter
    loop_n(Iter it, std::size_t count, F && f, CancelToken& tok)
    {
        typedef typename std::iterator_traits<Iter>::iterator_category cat;
        typedef typename hpx::traits::is_future<
            typename std::iterator_traits<Iter>::value_type
        >::type pred;

        return detail::loop_n<cat>::call(it, count, std::forward<F>(f), tok,
            pred());
    };

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position.
        template <typename IterCat>
        struct accumulate_n
        {
            template <typename Iter, typename T, typename Pred>
            static T call(Iter it, std::size_t count, T init, Pred && f,
                boost::mpl::false_)
            {
                for (/**/; count != 0; --count, ++it)
                    init = f(init, *it);
                return init;
            }
        };

        // specialization for random access iterators
        template <>
        struct accumulate_n<std::random_access_iterator_tag>
        {
            template <typename Iter, typename T, typename Pred>
            static T call(Iter it, std::size_t count, T init, Pred && f,
                boost::mpl::false_)
            {
                for (std::size_t i = 0; i != count; ++i)
                    init = f(init, it[i]);
                return init;
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename T, typename Pred>
    BOOST_FORCEINLINE Iter
    accumulate_n(Iter it, std::size_t count, T init, Pred && f)
    {
        typedef typename std::iterator_traits<Iter>::iterator_category cat;
        typedef typename hpx::traits::is_future<
            typename std::iterator_traits<Iter>::value_type
        >::type pred;

        return detail::accumulate_n<cat>::call(it, count, std::move(init),
            std::forward<F>(f), pred());
    }
}}}

#endif
