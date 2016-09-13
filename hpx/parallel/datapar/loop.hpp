//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2016 Matthias Kretz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_UTIL_LOOP_SEP_07_2016_1217PM)
#define HPX_PARALLEL_DATAPAR_UTIL_LOOP_SEP_07_2016_1217PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_VC_DATAPAR)
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/datapar/detail/iterator_helpers.hpp>
#include <hpx/parallel/datapar/execution_policy_fwd.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/util/decay.hpp>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>
#include <type_traits>

#include <Vc/Vc>

namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
//     template <typename F, typename Iter>
//     HPX_HOST_DEVICE HPX_FORCEINLINE
//     auto loop_step(parallel::v1::datapar_execution_policy, F && f,
//             Iter& it)
//     ->  decltype(
//             detail::datapar_loop_step<Iter>::callv(
//                 std::forward<F>(f), it)
//         )
//     {
//         return detail::datapar_loop_step<Iter>::callv(
//             std::forward<F>(f), it);
//     }
//
//     template <typename F, typename Iter>
//     HPX_HOST_DEVICE HPX_FORCEINLINE
//     auto loop_step(parallel::v1::datapar_task_execution_policy, F && f,
//             Iter& it)
//     ->  decltype(
//             detail::datapar_loop_step<Iter>::callv(
//                 std::forward<F>(f), it)
//             )
//     {
//         return detail::datapar_loop_step<Iter>::callv(
//             std::forward<F>(f), it);
//     }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Iter1, typename Iter2>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    auto loop_step(parallel::v1::datapar_execution_policy, F && f,
            Iter1& it1, Iter2& it2)
    ->  decltype(
            detail::datapar_loop_step2<Iter1, Iter2>::callv(
                std::forward<F>(f), it1, it2)
            )
    {
        return detail::datapar_loop_step2<Iter1, Iter2>::callv(
            std::forward<F>(f), it1, it2);
    }

    template <typename F, typename Iter1, typename Iter2>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    auto loop_step(parallel::v1::datapar_task_execution_policy, F && f,
            Iter1& it1, Iter2& it2)
    ->  decltype(
            detail::datapar_loop_step2<Iter1, Iter2>::callv(
                std::forward<F>(f), it1, it2)
            )
    {
        return detail::datapar_loop_step2<Iter1, Iter2>::callv(
            std::forward<F>(f), it1, it2);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Iter, typename Enable = void>
        struct loop_optimization
        {
            template <typename Iter1>
            static bool call(Iter1 const& first1, Iter1 const& last1)
            {
                return false;
            }
        };

        template <typename Iter>
        struct loop_optimization<Iter,
            typename std::enable_if<
                iterator_datapar_compatible<Iter>::value
            >::type>
        {
            template <typename Iter1>
            static bool call(Iter1 const& first1, Iter1 const& last1)
            {
                typedef typename std::iterator_traits<Iter1>::value_type
                    value_type;
                typedef Vc::Vector<value_type> V;

                return V::Size <= std::distance(first1, last1);
            }
        };
    }

    template <typename Iter>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    bool loop_optimization(parallel::v1::datapar_execution_policy,
        Iter const& first1, Iter const& last1)
    {
        return detail::loop_optimization<Iter>::call(first1, last1);
    }

    template <typename Iter>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    bool loop_optimization(parallel::v1::datapar_task_execution_policy,
        Iter const& first1, Iter const& last1)
    {
        return detail::loop_optimization<Iter>::call(first1, last1);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Abi>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        std::size_t count_bits(Vc::Mask<T, Abi> const& mask)
        {
            return mask.count();
        }

        template <typename F, typename Vector>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        typename hpx::util::decay<Vector>::type::EntryType
        accumulate_values(parallel::v1::datapar_execution_policy,
            F && f, Vector && value)
        {
            typedef typename hpx::util::decay<Vector>::type vector_type;
            typedef typename vector_type::EntryType entry_type;

            entry_type accum = value[0];
            for(size_t i = 1; i != value.size(); ++i)
            {
                accum = f(accum, entry_type(value[i]));
            }
            return accum;
        }

        template <typename F, typename Vector>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        typename hpx::util::decay<Vector>::type::EntryType
        accumulate_values(parallel::v1::datapar_task_execution_policy,
            F && f, Vector && value)
        {
            typedef typename hpx::util::decay<Vector>::type vector_type;
            typedef typename vector_type::EntryType entry_type;

            entry_type accum = value[0];
            for(size_t i = 1; i != value.size(); ++i)
            {
                accum = f(accum, entry_type(value[i]));
            }
            return accum;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Vector, typename T>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        T accumulate_values(parallel::v1::datapar_execution_policy,
            F && f, Vector && value, T accum)
        {
            for(size_t i = 0; i != value.size(); ++i)
            {
                accum = f(accum, T(value[i]));
            }
            return accum;
        }

        template <typename F, typename Vector, typename T>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        T accumulate_values(parallel::v1::datapar_task_execution_policy,
            F && f, Vector && value, T accum)
        {
            for(size_t i = 0; i != value.size(); ++i)
            {
                accum = f(accum, T(value[i]));
            }
            return accum;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin
    loop(ExPolicy&&, Begin begin, End end, F && f);

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function starting from a given
        // iterator position.
        template <typename Iterator>
        struct datapar_loop
        {
            typedef typename hpx::util::decay<Iterator>::type iterator_type;
            typedef typename std::iterator_traits<iterator_type>::value_type
                value_type;
            typedef Vc::Vector<value_type> V;

            ///////////////////////////////////////////////////////////////////
            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                iterator_datapar_compatible<Begin>::value, Begin
            >::type
            call(Begin first, End last, F && f)
            {
                while (data_alignment(first) && first != last)
                {
                    datapar_loop_step<Begin>::call1(f, first);
                }

                End const lastV = last - (V::Size + 1);
                while (first < lastV)
                {
                    datapar_loop_step<Begin>::callv(f, first);
                }

                while (first != last)
                {
                    datapar_loop_step<Begin>::call1(f, first);
                }

                return first;
            }

            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
               !iterator_datapar_compatible<Begin>::value, Begin
            >::type
            call(Begin it, End end, F && f)
            {
                return util::loop(parallel::v1::seq, it, end, std::forward<F>(f));
            }

//             template <typename Begin, typename End, typename CancelToken,
//                 typename F>
//             HPX_HOST_DEVICE HPX_FORCEINLINE
//             static typename std::enable_if<
//                 iterator_datapar_compatible<Begin>::value, Begin
//             >::type
//             call(Begin first, End last, CancelToken& tok, F && f)
//             {
//                 while (data_alignment(first) && first != last)
//                 {
//                     if (tok.was_cancelled())
//                         return first;
//                     datapar_loop_step<Begin>::call1(f, first);
//                 }
//
//                 End const lastV = last - (V::Size + 1);
//                 while (first < lastV)
//                 {
//                     if (tok.was_cancelled())
//                         return first;
//                     datapar_loop_step<Begin>::callv(f, first);
//                 }
//
//                 while (first != last)
//                 {
//                     if (tok.was_cancelled())
//                         break;
//                     datapar_loop_step<Begin>::call1(f, first);
//                 }
//
//                 return first;
//             }
//
//             template <typename Begin, typename End, typename CancelToken,
//                 typename F>
//             HPX_HOST_DEVICE HPX_FORCEINLINE
//             static typename std::enable_if<
//                !iterator_datapar_compatible<Begin>::value, Begin
//             >::type
//             call(Begin it, End end, CancelToken& tok, F && f)
//             {
//                 return util::loop(parallel::v1::seq, it, end, tok,
//                     std::forward<F>(f));
//             }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin
    loop(parallel::v1::datapar_execution_policy, Begin begin, End end, F && f)
    {
        return detail::datapar_loop<Begin>::call(begin, end, std::forward<F>(f));
    }

//     template <typename Begin, typename End, typename CancelToken, typename F>
//     HPX_HOST_DEVICE HPX_FORCEINLINE Begin
//     loop(parallel::v1::datapar_execution_policy, Begin begin, End end,
//         CancelToken& tok, F && f)
//     {
//         return detail::datapar_loop<Begin>::call(begin, end, tok,
//             std::forward<F>(f));
//     }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin
    loop(ExPolicy&&, Begin begin, End end, F && f);

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function starting from a given
        // iterator position.
        template <typename Iter1, typename Iter2>
        struct datapar_loop2
        {
            ///////////////////////////////////////////////////////////////////
            template <typename InIter1, typename InIter2, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                iterators_datapar_compatible<InIter1, InIter2>::value &&
                    iterator_datapar_compatible<InIter1>::value &&
                    iterator_datapar_compatible<InIter2>::value,
                std::pair<InIter1, InIter2>
            >::type
            call(InIter1 it1, InIter1 last1, InIter2 it2, F && f)
            {
                typedef typename hpx::util::decay<InIter1>::type iterator_type;
                typedef typename std::iterator_traits<iterator_type>::value_type
                    value_type;
                typedef Vc::Vector<value_type> V;

                if (detail::data_alignment(it1) || detail::data_alignment(it2))
                {
                    return std::make_pair(std::move(it1), std::move(it2));
                }

                InIter1 const last1V = last1 - (V::Size + 1);
                while (it1 < last1V)
                {
                    datapar_loop_step2<InIter1, InIter2>::callv(f, it1, it2);
                }

                return std::make_pair(std::move(it1), std::move(it2));
            }

            template <typename InIter1, typename InIter2, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                !iterators_datapar_compatible<InIter1, InIter2>::value ||
                    !iterator_datapar_compatible<InIter1>::value ||
                    !iterator_datapar_compatible<InIter2>::value,
                std::pair<InIter1, InIter2>
            >::type
            call(InIter1 it1, InIter1 last1, InIter2 it2, F && f)
            {
                return std::make_pair(std::move(it1), std::move(it2));
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter1, typename Iter2, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::pair<Iter1, Iter2>
    loop2(parallel::v1::datapar_execution_policy, Iter1 first1, Iter1 last1,
        Iter2 first2, F && f)
    {
        return detail::datapar_loop2<Iter1, Iter2>::call(first1, last1, first2,
            std::forward<F>(f));
    }

    template <typename Iter1, typename Iter2, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::pair<Iter1, Iter2>
    loop2(parallel::v1::datapar_task_execution_policy, Iter1 first1, Iter1 last1,
        Iter2 first2, F && f)
    {
        return detail::datapar_loop2<Iter1, Iter2>::call(first1, last1, first2,
            std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iter
    loop_n(ExPolicy &&, Iter it, std::size_t count, F && f);

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct datapar_loop_n
        {
            typedef typename hpx::util::decay<Iterator>::type iterator_type;
            typedef typename std::iterator_traits<iterator_type>::value_type
                value_type;
            typedef Vc::Vector<value_type> V;

            ///////////////////////////////////////////////////////////////
            template <typename InIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                iterator_datapar_compatible<InIter>::value, InIter
            >::type
            call(InIter first, std::size_t count, F && f)
            {
                std::size_t len = count;

                for (/* */; detail::data_alignment(first) && len != 0; --len)
                {
                    datapar_loop_step<InIter>::call1(f, first);
                }

                for (std::int64_t lenV = std::int64_t(count - (V::Size + 1));
                        lenV > 0; lenV -= V::Size, len -= V::Size)
                {
                    datapar_loop_step<InIter>::callv(f, first);
                }

                for (/* */; len != 0; --len)
                {
                    datapar_loop_step<InIter>::call1(f, first);
                }

                return first;
            }

            template <typename InIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                !iterator_datapar_compatible<InIter>::value, InIter
            >::type
            call(InIter first, std::size_t count, F && f)
            {
                return util::loop_n(parallel::v1::seq, first, count,
                    std::forward<F>(f));
            }

            ///////////////////////////////////////////////////////////////
            template <typename InIter, typename CancelToken, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                iterator_datapar_compatible<InIter>::value, InIter
            >::type
            call(InIter first, std::size_t count, CancelToken& tok, F && f)
            {
                std::size_t len = count;

                for (/* */; detail::data_alignment(first) && len != 0; --len)
                {
                    if (tok.was_cancelled())
                        return first;

                    datapar_loop_step<InIter>::call1(f, first);
                }

                for (std::int64_t lenV = std::int64_t(count - (V::Size + 1));
                        lenV > 0; lenV -= V::Size, len -= V::Size)
                {
                    if (tok.was_cancelled())
                        return first;

                    datapar_loop_step<InIter>::callv(f, first);
                }

                for (/* */; len != 0; --len)
                {
                    if (tok.was_cancelled())
                        return first;

                    datapar_loop_step<InIter>::call1(f, first);
                }

                return first;
            }

            template <typename InIter, typename CancelToken, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                !iterator_datapar_compatible<InIter>::value, InIter
            >::type
            call(InIter first, std::size_t count, CancelToken& tok, F && f)
            {
                return util::loop_n(parallel::v1::seq, first, count, tok,
                    std::forward<F>(f));
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iter
    loop_n(parallel::v1::datapar_execution_policy, Iter it,
        std::size_t count, F && f)
    {
        return detail::datapar_loop_n<Iter>::call(it, count, std::forward<F>(f));
    }

    template <typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iter
    loop_n(parallel::v1::datapar_task_execution_policy, Iter it,
        std::size_t count, F && f)
    {
        return detail::datapar_loop_n<Iter>::call(it, count, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename CancelToken, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iter
    loop_n(parallel::v1::datapar_execution_policy, Iter it,
        std::size_t count, CancelToken& tok, F && f)
    {
        return detail::datapar_loop_n<Iter>::call(it, count, tok,
            std::forward<F>(f));
    }

    template <typename Iter, typename CancelToken, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iter
    loop_n(parallel::v1::datapar_task_execution_policy, Iter it,
        std::size_t count, CancelToken& tok, F && f)
    {
        return detail::datapar_loop_n<Iter>::call(it, count, tok,
            std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
//     namespace detail
//     {
//         ///////////////////////////////////////////////////////////////////////
//         // Helper class to repeatedly call a function starting from a given
//         // iterator position.
//         template <typename Iter1, typename Iter2>
//         struct datapar_loop2_n
//         {
//             ///////////////////////////////////////////////////////////////
//             template <typename InIter1, typename InIter2, typename F>
//             HPX_HOST_DEVICE HPX_FORCEINLINE
//             static typename std::enable_if<
//                 iterators_datapar_compatible<InIter1, InIter2>::value &&
//                     iterator_datapar_compatible<InIter1>::value &&
//                     iterator_datapar_compatible<InIter2>::value,
//                 std::pair<InIter1, InIter2>
//             >::type
//             call(InIter1 it1, std::size_t count, InIter2 it2, F && f)
//             {
//                 typedef typename hpx::util::decay<InIter1>::type iterator_type;
//                 typedef typename std::iterator_traits<iterator_type>::value_type
//                     value_type;
//                 typedef Vc::Vector<value_type> V;
//
//                 if (detail::data_alignment(it1) || detail::data_alignment(it2))
//                 {
//                     return std::make_pair(std::move(it1), std::move(it2));
//                 }
//
//                 for (std::int64_t lenV = std::int64_t(count - (V::Size + 1));
//                         lenV > 0; lenV -= V::Size)
//                 {
//                     datapar_loop_step2<InIter1, InIter2>::callv(f, it1, it2);
//                 }
//
//                 return std::make_pair(std::move(it1), std::move(it2));
//             }
//
//             template <typename InIter1, typename InIter2, typename F>
//             HPX_HOST_DEVICE HPX_FORCEINLINE
//             static typename std::enable_if<
//                 !iterators_datapar_compatible<InIter1, InIter2>::value ||
//                     !iterator_datapar_compatible<InIter1>::value ||
//                     !iterator_datapar_compatible<InIter2>::value,
//                 std::pair<InIter1, InIter2>
//             >::type
//             call(InIter1 it1, std::size_t count, InIter2 it2, F && f)
//             {
//                 return std::make_pair(std::move(it1), std::move(it2));
//             }
//         };
//     }
//
//     template <typename Begin1, typename Begin2, typename F>
//     HPX_HOST_DEVICE HPX_FORCEINLINE std::pair<Begin1, Begin2>
//     loop2_n(parallel::v1::datapar_execution_policy,
//         Begin1 begin1, std::size_t count, Begin2 begin2, F && f)
//     {
//         return detail::datapar_loop2_n<Begin1, Begin2>::call(begin1, count,
//             begin2, std::forward<F>(f));
//     }
//
//     template <typename Begin1, typename Begin2, typename F>
//     HPX_HOST_DEVICE HPX_FORCEINLINE std::pair<Begin1, Begin2>
//     loop2_n(parallel::v1::datapar_task_execution_policy,
//         Begin1 begin1, std::size_t count, Begin2 begin2, F && f)
//     {
//         return detail::datapar_loop2_n<Begin1, Begin2>::call(begin1, count,
//             begin2, std::forward<F>(f));
//     }
}}}

#endif
#endif

