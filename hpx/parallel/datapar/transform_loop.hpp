//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2016 Matthias Kretz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_TRANSFORM_LOOP_SEP_08_2016_0657PM)
#define HPX_PARALLEL_DATAPAR_TRANSFORM_LOOP_SEP_08_2016_0657PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_VC_DATAPAR)
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/tuple.hpp>

#include <hpx/parallel/datapar/detail/iterator_helpers.hpp>
#include <hpx/parallel/datapar/execution_policy_fwd.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/traits/is_iterator.hpp>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

#include <Vc/Vc>

namespace hpx { namespace parallel { namespace util
{
    template <typename F, typename InIter, typename OutIter>
    HPX_HOST_DEVICE HPX_FORCEINLINE void
    transform_loop_step(parallel::v1::datapar_execution_policy, F && f,
        InIter it, OutIter dest)
    {
        detail::datapar_transform_loop_step::call1(std::forward<F>(f),
            it, dest, Vc::Unaligned);
    }

    template <typename F, typename InIter1, typename InIter2, typename OutIter>
    HPX_HOST_DEVICE HPX_FORCEINLINE void
    transform_loop_step(parallel::v1::datapar_execution_policy, F && f,
        InIter1 it1, InIter2 it2, OutIter dest)
    {
        detail::datapar_transform_loop_step::call1(std::forward<F>(f),
            it1, it2, dest, Vc::Unaligned);
    }

    template <typename F, typename InIter, typename OutIter>
    HPX_HOST_DEVICE HPX_FORCEINLINE void
    transform_loop_step(parallel::v1::datapar_task_execution_policy, F && f,
        InIter it, OutIter dest)
    {
        detail::datapar_transform_loop_step::call1(std::forward<F>(f),
            it, dest, Vc::Unaligned);
    }

    template <typename F, typename InIter1, typename InIter2, typename OutIter>
    HPX_HOST_DEVICE HPX_FORCEINLINE void
    transform_loop_step(parallel::v1::datapar_task_execution_policy, F && f,
        InIter1 it1, InIter2 it2, OutIter dest)
    {
        detail::datapar_transform_loop_step::call1(std::forward<F>(f),
            it1, it2, dest, Vc::Unaligned);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Iterator>
        struct datapar_transform_loop_n
        {
            typedef typename hpx::util::decay<Iterator>::type iterator_type;
            typedef Vc::Vector<typename iterator_type::value_type> V;

            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                std::is_arithmetic<typename InIter::value_type>::value &&
                    hpx::traits::is_random_access_iterator<InIter>::value &&
                    hpx::traits::is_random_access_iterator<OutIter>::value &&
                    iterators_datapar_compatible<InIter, OutIter>::value,
                std::pair<InIter, OutIter>
            >::type
            call(InIter first, std::size_t count, OutIter dest, F && f)
            {
                std::size_t len = count;

                // fall back to unaligned execution if input and output types
                // are not compatible
                if (data_alignment(first) != data_alignment(dest))
                {
                    for (std::int64_t lenV = std::int64_t(count - (V::Size + 1));
                            lenV > 0;
                            lenV -= V::Size, len -= V::Size, first += V::Size,
                                dest += V::Size)
                    {
                        datapar_transform_loop_step::callv(f, first, dest,
                            Vc::Unaligned);
                    }

                    for (/* */; len != 0; (void) --len, ++first, ++dest)
                    {
                        datapar_transform_loop_step::call1(f, first, dest,
                            Vc::Unaligned);
                    }
                }
                else
                {
                    for (/* */; data_alignment(first) && len != 0;
                         (void) --len, ++first, ++dest)
                    {
                        datapar_transform_loop_step::call1(f, first, dest,
                            Vc::Aligned);
                    }

                    for (std::int64_t lenV = std::int64_t(count - (V::Size + 1));
                            lenV > 0;
                            lenV -= V::Size, len -= V::Size, first += V::Size,
                                dest += V::Size)
                    {
                        datapar_transform_loop_step::callv(f, first, dest,
                            Vc::Aligned);
                    }

                    for (/* */; len != 0; (void) --len, ++first, ++dest)
                    {
                        datapar_transform_loop_step::call1(f, first, dest,
                            Vc::Aligned);
                    }
                }
                return std::make_pair(std::move(first), std::move(dest));
            }

            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
               !std::is_arithmetic<typename InIter::value_type>::value ||
                   !hpx::traits::is_random_access_iterator<InIter>::value ||
                   !hpx::traits::is_random_access_iterator<OutIter>::value ||
                   !iterators_datapar_compatible<InIter, OutIter>::value ||
                   !iterators_datapar_compatible<InIter, OutIter>::value,
                std::pair<InIter, OutIter>
            >::type
            call(InIter first, std::size_t count, OutIter dest, F && f)
            {
                return util::transform_loop_n(parallel::v1::seq, first, count,
                    dest, std::forward<F>(f));
            }
        };
    }

    template <typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::pair<Iter, OutIter>
    transform_loop_n(parallel::v1::datapar_execution_policy, Iter it,
        std::size_t count, OutIter dest, F && f)
    {
        return detail::datapar_transform_loop_n<Iter>::call(it, count, dest,
            std::forward<F>(f));
    }

    template <typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::pair<Iter, OutIter>
    transform_loop_n(parallel::v1::datapar_task_execution_policy, Iter it,
        std::size_t count, OutIter dest, F && f)
    {
        return detail::datapar_transform_loop_n<Iter>::call(it, count, dest,
            std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Iterator>
        struct datapar_transform_loop
        {
            typedef typename hpx::util::decay<Iterator>::type iterator_type;

            typedef Vc::Vector<typename iterator_type::value_type> V;
            typedef Vc::Scalar::Vector<typename iterator_type::value_type> V1;

            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                std::is_arithmetic<typename InIter::value_type>::value &&
                    hpx::traits::is_random_access_iterator<InIter>::value &&
                    hpx::traits::is_random_access_iterator<OutIter>::value &&
                    iterators_datapar_compatible<InIter, OutIter>::value,
                std::pair<InIter, OutIter>
            >::type
            call(InIter first, InIter last, OutIter dest, F && f)
            {
                return util::transform_loop_n(parallel::v1::datapar_execution,
                    first, std::distance(first, last), dest, std::forward<F>(f));
            }

            template <typename InIter, typename OutIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
               !std::is_arithmetic<typename InIter::value_type>::value ||
                   !hpx::traits::is_random_access_iterator<InIter>::value ||
                   !hpx::traits::is_random_access_iterator<OutIter>::value ||
                   !iterators_datapar_compatible<InIter, OutIter>::value,
                std::pair<InIter, OutIter>
            >::type
            call(InIter first, InIter last, OutIter dest, F && f)
            {
                return util::transform_loop(parallel::v1::seq,
                    first, last, dest, std::forward<F>(f));
            }
        };
    }

    template <typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::pair<Iter, OutIter>
    transform_loop(parallel::v1::datapar_execution_policy, Iter it, Iter end,
        OutIter dest, F && f)
    {
        return detail::datapar_transform_loop<Iter>::call(it, end, dest,
            std::forward<F>(f));
    }

    template <typename Iter, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    std::pair<Iter, OutIter>
    transform_loop(parallel::v1::datapar_task_execution_policy, Iter it,
        Iter end, OutIter dest, F && f)
    {
        return detail::datapar_transform_loop<Iter>::call(it, end, dest,
            std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Iter1, typename Iter2>
        struct datapar_transform_binary_loop_n
        {
            typedef typename hpx::util::decay<Iter1>::type iterator1_type;
            typedef Vc::Vector<typename iterator1_type::value_type> V;

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                std::is_arithmetic<typename InIter1::value_type>::value &&
                    std::is_arithmetic<typename InIter2::value_type>::value &&
                    hpx::traits::is_random_access_iterator<InIter1>::value &&
                    hpx::traits::is_random_access_iterator<InIter2>::value &&
                    hpx::traits::is_random_access_iterator<OutIter>::value &&
                    iterators_datapar_compatible<InIter1, OutIter>::value &&
                    iterators_datapar_compatible<InIter2, OutIter>::value,
                hpx::util::tuple<InIter1, InIter2, OutIter>
            >::type
            call(InIter1 first1, std::size_t count, InIter2 first2,
                OutIter dest, F && f)
            {
                std::size_t len = count;

                // fall back to unaligned execution if input and output types
                // are not compatible
                if (data_alignment(first1) != data_alignment(dest) ||
                    data_alignment(first2) != data_alignment(dest))
                {
                    for (std::int64_t lenV = std::int64_t(count - (V::Size + 1));
                            lenV > 0;
                            lenV -= V::Size, len -= V::Size,
                                first1 += V::Size, first2 += V::Size,
                                dest += V::Size)
                    {
                        datapar_transform_loop_step::callv(f, first1, first2,
                            dest, Vc::Unaligned);
                    }

                    for (/* */; len != 0;
                            (void) --len, ++first1, ++first2, ++dest)
                    {
                        datapar_transform_loop_step::call1(f, first1, first2,
                            dest, Vc::Unaligned);
                    }
                }
                else
                {
                    for (/* */; data_alignment(first1) && len != 0;
                            (void) --len, ++first1, ++first2, ++dest)
                    {
                        datapar_transform_loop_step::call1(f, first1, first2,
                            dest, Vc::Aligned);
                    }

                    for (std::int64_t lenV = std::int64_t(count - (V::Size + 1));
                            lenV > 0;
                            lenV -= V::Size, len -= V::Size,
                                first1 += V::Size, first2 += V::Size,
                                dest += V::Size)
                    {
                        datapar_transform_loop_step::callv(f, first1, first2,
                            dest, Vc::Aligned);
                    }

                    for (/* */; len != 0;
                            (void) --len, ++first1, ++first2, ++dest)
                    {
                        datapar_transform_loop_step::call1(f, first1, first2,
                            dest, Vc::Aligned);
                    }
                }
                return hpx::util::make_tuple(std::move(first1),
                    std::move(first2), std::move(dest));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
               !std::is_arithmetic<typename InIter1::value_type>::value ||
                   !std::is_arithmetic<typename InIter2::value_type>::value ||
                   !hpx::traits::is_random_access_iterator<InIter1>::value ||
                   !hpx::traits::is_random_access_iterator<InIter2>::value ||
                   !hpx::traits::is_random_access_iterator<OutIter>::value ||
                   !iterators_datapar_compatible<InIter1, OutIter>::value ||
                   !iterators_datapar_compatible<InIter2, OutIter>::value,
                hpx::util::tuple<InIter1, InIter2, OutIter>
            >::type
            call(InIter1 first1, std::size_t count, InIter2 first2,
                OutIter dest, F && f)
            {
                return util::transform_binary_loop_n(parallel::v1::seq,
                    first1, count, first2, dest, std::forward<F>(f));
            }
        };
    }

    template <typename InIter1, typename InIter2, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    hpx::util::tuple<InIter1, InIter2, OutIter>
    transform_binary_loop_n(parallel::v1::datapar_execution_policy,
        InIter1 first1, std::size_t count, InIter2 first2, OutIter dest, F && f)
    {
        return detail::datapar_transform_binary_loop_n<InIter1, InIter2>::call(
            first1, count, first2, dest, std::forward<F>(f));
    }

    template <typename InIter1, typename InIter2, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    hpx::util::tuple<InIter1, InIter2, OutIter>
    transform_binary_loop_n(parallel::v1::datapar_task_execution_policy,
        InIter1 first1, std::size_t count, InIter2 first2, OutIter dest, F && f)
    {
        return detail::datapar_transform_binary_loop_n<InIter1, InIter2>::call(
            first1, count, first2, dest, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Iter1, typename Iter2>
        struct datapar_transform_binary_loop
        {
            typedef typename hpx::util::decay<Iter1>::type iterator1_type;
            typedef Vc::Scalar::Vector<typename iterator1_type::value_type> V11;

            typedef typename hpx::util::decay<Iter2>::type iterator2_type;
            typedef Vc::Scalar::Vector<typename iterator2_type::value_type> V12;

            typedef Vc::Vector<typename iterator1_type::value_type> V1;
            typedef Vc::Vector<typename iterator2_type::value_type> V2;

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                std::is_arithmetic<typename InIter1::value_type>::value &&
                    std::is_arithmetic<typename InIter2::value_type>::value &&
                    hpx::traits::is_random_access_iterator<InIter1>::value &&
                    hpx::traits::is_random_access_iterator<InIter2>::value &&
                    hpx::traits::is_random_access_iterator<OutIter>::value &&
                    iterators_datapar_compatible<InIter1, OutIter>::value &&
                    iterators_datapar_compatible<InIter2, OutIter>::value,
                hpx::util::tuple<InIter1, InIter2, OutIter>
            >::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest,
                F && f)
            {
                return util::transform_binary_loop_n(
                    parallel::v1::datapar_execution,
                    first1, std::distance(first1, last1), first2, dest,
                    std::forward<F>(f));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
               !std::is_arithmetic<typename InIter1::value_type>::value ||
                   !std::is_arithmetic<typename InIter2::value_type>::value ||
                   !hpx::traits::is_random_access_iterator<InIter1>::value ||
                   !hpx::traits::is_random_access_iterator<InIter2>::value ||
                   !hpx::traits::is_random_access_iterator<OutIter>::value ||
                   !iterators_datapar_compatible<InIter1, OutIter>::value ||
                   !iterators_datapar_compatible<InIter2, OutIter>::value,
                hpx::util::tuple<InIter1, InIter2, OutIter>
            >::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest,
                F && f)
            {
                return util::transform_binary_loop(parallel::v1::seq,
                    first1, last1, first2, dest, std::forward<F>(f));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                std::is_arithmetic<typename InIter1::value_type>::value &&
                    std::is_arithmetic<typename InIter2::value_type>::value &&
                    hpx::traits::is_random_access_iterator<InIter1>::value &&
                    hpx::traits::is_random_access_iterator<InIter2>::value &&
                    hpx::traits::is_random_access_iterator<OutIter>::value &&
                    iterators_datapar_compatible<InIter1, OutIter>::value &&
                    iterators_datapar_compatible<InIter2, OutIter>::value,
                hpx::util::tuple<InIter1, InIter2, OutIter>
            >::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
                OutIter dest, F && f)
            {
                std::size_t count = (std::min)(std::distance(first1, last1),
                    std::distance(first2, last2));

                return util::transform_binary_loop_n(
                    parallel::v1::datapar_execution,
                    first1, count, first2, dest, std::forward<F>(f));
            }

            template <typename InIter1, typename InIter2, typename OutIter,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
               !std::is_arithmetic<typename InIter1::value_type>::value ||
                   !std::is_arithmetic<typename InIter2::value_type>::value ||
                   !hpx::traits::is_random_access_iterator<InIter1>::value ||
                   !hpx::traits::is_random_access_iterator<InIter2>::value ||
                   !hpx::traits::is_random_access_iterator<OutIter>::value ||
                   !iterators_datapar_compatible<InIter1, OutIter>::value ||
                   !iterators_datapar_compatible<InIter2, OutIter>::value,
                hpx::util::tuple<InIter1, InIter2, OutIter>
            >::type
            call(InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
                OutIter dest, F && f)
            {
                return util::transform_binary_loop(parallel::v1::seq,
                    first1, last1, first2, last2, dest, std::forward<F>(f));
            }
        };
    }

    template <typename InIter1, typename InIter2, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    hpx::util::tuple<InIter1, InIter2, OutIter>
    transform_binary_loop(parallel::v1::datapar_execution_policy,
        InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest, F && f)
    {
        return detail::datapar_transform_binary_loop<InIter1, InIter2>::call(
            first1, last1, first2, dest, std::forward<F>(f));
    }

    template <typename InIter1, typename InIter2, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    hpx::util::tuple<InIter1, InIter2, OutIter>
    transform_binary_loop(parallel::v1::datapar_task_execution_policy,
        InIter1 first1, InIter1 last1, InIter2 first2, OutIter dest, F && f)
    {
        return detail::datapar_transform_binary_loop<InIter1, InIter2>::call(
            first1, last1, first2, dest, std::forward<F>(f));
    }

    template <typename InIter1, typename InIter2, typename OutIter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    hpx::util::tuple<InIter1, InIter2, OutIter>
    transform_binary_loop(parallel::v1::datapar_execution_policy,
        InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
        OutIter dest, F && f)
    {
        return detail::datapar_transform_binary_loop<InIter1, InIter2>::call(
            first1, last1, first2, last2, dest, std::forward<F>(f));
    }

    template <typename InIter1, typename InIter2,
        typename OutIter, typename F, typename Proj1, typename Proj2>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    hpx::util::tuple<InIter1, InIter2, OutIter>
    transform_binary_loop(parallel::v1::datapar_task_execution_policy,
        InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
        OutIter dest, F && f)
    {
        return detail::datapar_transform_binary_loop<InIter1, InIter2>::call(
            first1, last1, first2, last2, dest, std::forward<F>(f));
    }
}}}

#endif
#endif

