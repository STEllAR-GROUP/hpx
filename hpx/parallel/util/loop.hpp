//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_LOOP_MAY_27_2014_1040PM)
#define HPX_PARALLEL_UTIL_LOOP_MAY_27_2014_1040PM

#include <hpx/hpx_fwd.hpp>

#include <iterator>
#include <algorithm>

#include <boost/atomic.hpp>

namespace hpx { namespace parallel { namespace util
{
    namespace detail
    {
        struct no_data
        {
            bool operator<= (no_data) const { return true; }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // cancellation_token is used for premature cancellation of algorithms
    template <typename T = detail::no_data, typename Pred = std::less_equal<T> >
    class cancellation_token
    {
    private:
        typedef boost::atomic<T> flag_type;
        boost::shared_ptr<flag_type> was_cancelled_;

    public:
        cancellation_token(T data)
          : was_cancelled_(boost::make_shared<flag_type>(data))
        {}

        bool was_cancelled(T data) const BOOST_NOEXCEPT
        {
            return Pred()(was_cancelled_->load(boost::memory_order_relaxed), data);
        }

        void cancel(T data) BOOST_NOEXCEPT
        {
            T old_data = was_cancelled_->load(boost::memory_order_relaxed);

            do {
                if (Pred()(old_data, data))
                    break;      // if we already have a closer one, break

            } while (!was_cancelled_->compare_exchange_strong(old_data, data,
                boost::memory_order_relaxed));
        }

        T get_data() const BOOST_NOEXCEPT
        {
            return was_cancelled_->load(boost::memory_order_relaxed);
        }
    };

    // special case for when no additional data needs to be stored at the
    // cancellation point
    template <>
    class cancellation_token<detail::no_data, std::less_equal<detail::no_data> >
    {
    private:
        typedef boost::atomic<bool> flag_type;
        boost::shared_ptr<flag_type> was_cancelled_;

    public:
        cancellation_token()
          : was_cancelled_(boost::make_shared<flag_type>(false))
        {}

        bool was_cancelled() const BOOST_NOEXCEPT
        {
            return was_cancelled_->load(boost::memory_order_relaxed);
        }

        void cancel() BOOST_NOEXCEPT
        {
            was_cancelled_->store(true, boost::memory_order_relaxed);
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
            static Iter call(Iter it, Iter end, F && f)
            {
                for (/**/; it != end; ++it)
                    f(*it);

                return it;
            }

            template <typename Iter, typename CancelToken, typename F>
            static Iter call(Iter it, Iter end, CancelToken& tok, F && func)
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
    }

    ///////////////////////////////////////////////////////////////////////////
    // no special handling for futures
    template <typename Iter, typename F>
    BOOST_FORCEINLINE Iter
    loop(Iter begin, Iter end, F && f)
    {
        typedef typename std::iterator_traits<Iter>::iterator_category cat;
        return detail::loop<cat>::call(begin, end, std::forward<F>(f));
    }

    template <typename Iter, typename CancelToken, typename F>
    BOOST_FORCEINLINE Iter
    loop(Iter begin, Iter end, CancelToken& tok, F && f)
    {
        typedef typename std::iterator_traits<Iter>::iterator_category cat;
        return detail::loop<cat>::call(begin, end, tok, std::forward<F>(f));
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position.
        template <typename IterCat>
        struct loop_n
        {
            ///////////////////////////////////////////////////////////////////
            // handle sequences of non-futures
            template <typename Iter, typename F>
            static Iter call(Iter it, std::size_t count, F && f)
            {
                for (/**/; count != 0; --count, ++it)
                    f(*it);
                return it;
            }

            template <typename Iter, typename CancelToken, typename F>
            static Iter call(Iter it, std::size_t count, CancelToken& tok,
                F && f)
            {
                for (/**/; count != 0; --count, ++it)
                {
                    f(*it);
                    if (tok.was_cancelled())
                        break;
                }
                return it;
            }
        };

        // specialization for random access iterators
//         template <>
//         struct loop_n<std::random_access_iterator_tag>
//         {
//             ///////////////////////////////////////////////////////////////////
//             // handle sequences of non-futures
//             template <typename Iter, typename F>
//             static Iter call(Iter it, std::size_t count, F && f)
//             {
//                 for (std::size_t i = 0; i != count; ++i)
//                     f(it[i]);
//
//                 std::advance(it, count);
//                 return it;
//             }
//
//             template <typename Iter, typename CancelToken, typename F>
//             static Iter call(Iter it, std::size_t count, CancelToken& tok,
//                 F && func)
//             {
//                 std::size_t i = 0;
//                 for (/**/; i != count; ++i)
//                 {
//                     func(it[i]);
//                     if (tok.was_cancelled())
//                         break;
//                 }
//                 std::advance(it, i);
//                 return it;
//             }
//         };
    }

    ///////////////////////////////////////////////////////////////////////////
    // no special handling of futures
    template <typename Iter, typename F>
    BOOST_FORCEINLINE Iter
    loop_n(Iter it, std::size_t count, F && f)
    {
        typedef typename std::iterator_traits<Iter>::iterator_category cat;
        return detail::loop_n<cat>::call(it, count, std::forward<F>(f));
    }

    template <typename Iter, typename CancelToken, typename F>
    BOOST_FORCEINLINE Iter
    loop_n(Iter it, std::size_t count, CancelToken& tok, F && f)
    {
        typedef typename std::iterator_traits<Iter>::iterator_category cat;
        return detail::loop_n<cat>::call(it, count, tok, std::forward<F>(f));
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position.
        template <typename IterCat>
        struct loop_idx_n
        {
            ///////////////////////////////////////////////////////////////////
            // handle sequences of non-futures
            template <typename Iter, typename F>
            static Iter
            call(std::size_t base_idx, Iter it, std::size_t count, F && f)
            {
                for (/**/; count != 0; --count, ++it, ++base_idx)
                    f(*it, base_idx);

                return it;
            }

            template <typename Iter, typename CancelToken, typename F>
            static Iter
            call(std::size_t base_idx, Iter it, std::size_t count,
                CancelToken& tok, F && f)
            {
                for (/**/; count != 0; --count, ++it, ++base_idx)
                {
                    f(*it, base_idx);
                    if (tok.was_cancelled(base_idx))
                        break;
                }
                return it;
            }
        };

        // specialization for random access iterators
        template <>
        struct loop_idx_n<std::random_access_iterator_tag>
        {
            ///////////////////////////////////////////////////////////////////
            // handle sequences of non-futures
            template <typename Iter, typename F>
            static Iter
            call(std::size_t base_idx, Iter it, std::size_t count, F && f)
            {
                for (std::size_t i = 0; i != count; ++i, ++base_idx)
                    f(it[i], base_idx);

                std::advance(it, count);
                return it;
            }

            template <typename Iter, typename CancelToken, typename F>
            static Iter call(std::size_t base_idx, Iter it, std::size_t count,
                CancelToken& tok, F && func)
            {
                std::size_t i = 0;
                for (/**/; i != count; ++i, ++base_idx)
                {
                    func(it[i], base_idx);
                    if (tok.was_cancelled(base_idx))
                        break;
                }
                std::advance(it, i);
                return it;
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // no special handling of futures
    template <typename Iter, typename F>
    BOOST_FORCEINLINE Iter
    loop_idx_n(std::size_t base_idx, Iter it, std::size_t count, F && f)
    {
        typedef typename std::iterator_traits<Iter>::iterator_category cat;
        return detail::loop_idx_n<cat>::call(base_idx, it, count,
            std::forward<F>(f));
    }

    template <typename Iter, typename CancelToken, typename F>
    BOOST_FORCEINLINE Iter
    loop_idx_n(std::size_t base_idx, Iter it, std::size_t count,
        CancelToken& tok, F && f)
    {
        typedef typename std::iterator_traits<Iter>::iterator_category cat;
        return detail::loop_idx_n<cat>::call(base_idx, it, count, tok,
            std::forward<F>(f));
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position.
        template <typename IterCat>
        struct accumulate_n
        {
            template <typename Iter, typename T, typename Pred>
            static T call(Iter it, std::size_t count, T init, Pred && f)
            {
                for (/**/; count != 0; --count, ++it)
                    init = f(init, *it);
                return init;
            }
        };

//         // specialization for random access iterators
//         template <>
//         struct accumulate_n<std::random_access_iterator_tag>
//         {
//             template <typename Iter, typename T, typename Pred>
//             static T call(Iter it, std::size_t count, T init, Pred && f)
//             {
//                 for (std::size_t i = 0; i != count; ++i)
//                     init = f(init, it[i]);
//                 return init;
//             }
//         };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename T, typename Pred>
    BOOST_FORCEINLINE T
    accumulate_n(Iter it, std::size_t count, T init, Pred && f)
    {
        typedef typename std::iterator_traits<Iter>::iterator_category cat;
        return detail::accumulate_n<cat>::call(it, count, std::move(init),
            std::forward<Pred>(f));
    }
}}}

#endif
