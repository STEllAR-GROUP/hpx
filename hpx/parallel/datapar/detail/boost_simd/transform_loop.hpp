//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_BOOST_SIMD_TRANSFORM_LOOP_SEP_22_2016_0224PM)
#define HPX_PARALLEL_DATAPAR_BOOST_SIMD_TRANSFORM_LOOP_SEP_22_2016_0224PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_BOOST_SIMD)
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/tuple.hpp>

#include <hpx/parallel/datapar/execution_policy_fwd.hpp>
#include <hpx/parallel/datapar/iterator_helpers.hpp>
#include <hpx/parallel/datapar/transform_loop_fwd.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/traits/is_iterator.hpp>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

#include <boost/simd.hpp>

namespace hpx { namespace parallel { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    struct datapar_transform_loop_n
    {
        typedef typename hpx::util::decay<Iterator>::type iterator_type;
        typedef boost::simd::pack<typename iterator_type::value_type> V;

        template <typename InIter, typename OutIter, typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static typename std::enable_if<
            iterators_datapar_compatible<InIter, OutIter>::value &&
                iterator_datapar_compatible<InIter>::value &&
                iterator_datapar_compatible<OutIter>::value,
            std::pair<InIter, OutIter>
        >::type
        call(InIter first, std::size_t count, OutIter dest, F && f)
        {
            std::size_t len = count;

            for (/* */; data_alignment(first) && len != 0; --len)
            {
                datapar_transform_loop_step::call1(f, first, dest);
            }

            for (std::int64_t lenV = std::int64_t(count - (V::static_size + 1));
                    lenV > 0; lenV -= V::static_size, len -= V::static_size)
            {
                datapar_transform_loop_step::callv(f, first, dest);
            }

            for (/* */; len != 0; --len)
            {
                datapar_transform_loop_step::call1(f, first, dest);
            }

            return std::make_pair(std::move(first), std::move(dest));
        }

        template <typename InIter, typename OutIter, typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static typename std::enable_if<
            !iterators_datapar_compatible<InIter, OutIter>::value ||
                !iterator_datapar_compatible<InIter>::value ||
                !iterator_datapar_compatible<OutIter>::value,
            std::pair<InIter, OutIter>
        >::type
        call(InIter first, std::size_t count, OutIter dest, F && f)
        {
            return util::transform_loop_n(parallel::v1::seq, first, count,
                dest, std::forward<F>(f));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    struct datapar_transform_loop
    {
        typedef typename hpx::util::decay<Iterator>::type iterator_type;

        typedef boost::simd::pack<typename iterator_type::value_type> V;
        typedef boost::simd::pack<typename iterator_type::value_type, 1> V1;

        template <typename InIter, typename OutIter, typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static typename std::enable_if<
            iterators_datapar_compatible<InIter, OutIter>::value &&
                iterator_datapar_compatible<InIter>::value &&
                iterator_datapar_compatible<OutIter>::value,
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
            !iterators_datapar_compatible<InIter, OutIter>::value ||
                !iterator_datapar_compatible<InIter>::value ||
                !iterator_datapar_compatible<OutIter>::value,
            std::pair<InIter, OutIter>
        >::type
        call(InIter first, InIter last, OutIter dest, F && f)
        {
            return util::transform_loop(parallel::v1::seq,
                first, last, dest, std::forward<F>(f));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter1, typename Iter2>
    struct datapar_transform_binary_loop_n
    {
        typedef typename hpx::util::decay<Iter1>::type iterator1_type;
        typedef boost::simd::pack<typename iterator1_type::value_type> V;

        template <typename InIter1, typename InIter2, typename OutIter,
            typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static typename std::enable_if<
            iterators_datapar_compatible<InIter1, OutIter>::value &&
                iterators_datapar_compatible<InIter2, OutIter>::value &&
                iterator_datapar_compatible<InIter1>::value &&
                iterator_datapar_compatible<InIter2>::value &&
                iterator_datapar_compatible<OutIter>::value,
            hpx::util::tuple<InIter1, InIter2, OutIter>
        >::type
        call(InIter1 first1, std::size_t count, InIter2 first2,
            OutIter dest, F && f)
        {
            std::size_t len = count;

            for (/* */; data_alignment(first1) && len != 0; --len)
            {
                datapar_transform_loop_step::call1(f, first1, first2, dest);
            }

            for (std::int64_t lenV = std::int64_t(count - (V::static_size + 1));
                    lenV > 0; lenV -= V::static_size, len -= V::static_size)
            {
                datapar_transform_loop_step::callv(f, first1, first2, dest);
            }

            for (/* */; len != 0; --len)
            {
                datapar_transform_loop_step::call1(f, first1, first2, dest);
            }

            return hpx::util::make_tuple(std::move(first1),
                std::move(first2), std::move(dest));
        }

        template <typename InIter1, typename InIter2, typename OutIter,
            typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static typename std::enable_if<
            !iterators_datapar_compatible<InIter1, OutIter>::value ||
                !iterators_datapar_compatible<InIter2, OutIter>::value ||
                !iterator_datapar_compatible<InIter1>::value ||
                !iterator_datapar_compatible<InIter2>::value ||
                !iterator_datapar_compatible<OutIter>::value,
            hpx::util::tuple<InIter1, InIter2, OutIter>
        >::type
        call(InIter1 first1, std::size_t count, InIter2 first2,
            OutIter dest, F && f)
        {
            return util::transform_binary_loop_n(parallel::v1::seq,
                first1, count, first2, dest, std::forward<F>(f));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter1, typename Iter2>
    struct datapar_transform_binary_loop
    {
        typedef typename hpx::util::decay<Iter1>::type iterator1_type;
        typedef boost::simd::pack<typename iterator1_type::value_type, 1> V11;

        typedef typename hpx::util::decay<Iter2>::type iterator2_type;
        typedef boost::simd::pack<typename iterator2_type::value_type, 1> V12;

        typedef boost::simd::pack<typename iterator1_type::value_type> V1;
        typedef boost::simd::pack<typename iterator2_type::value_type> V2;

        template <typename InIter1, typename InIter2, typename OutIter,
            typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static typename std::enable_if<
            iterators_datapar_compatible<InIter1, OutIter>::value &&
                iterators_datapar_compatible<InIter2, OutIter>::value &&
                iterator_datapar_compatible<InIter1>::value &&
                iterator_datapar_compatible<InIter2>::value &&
                iterator_datapar_compatible<OutIter>::value,
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
            !iterators_datapar_compatible<InIter1, OutIter>::value ||
                !iterators_datapar_compatible<InIter2, OutIter>::value ||
                !iterator_datapar_compatible<InIter1>::value ||
                !iterator_datapar_compatible<InIter2>::value ||
                !iterator_datapar_compatible<OutIter>::value,
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
            iterators_datapar_compatible<InIter1, OutIter>::value &&
                iterators_datapar_compatible<InIter2, OutIter>::value &&
                iterator_datapar_compatible<InIter1>::value &&
                iterator_datapar_compatible<InIter2>::value &&
                iterator_datapar_compatible<OutIter>::value,
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
            !iterators_datapar_compatible<InIter1, OutIter>::value ||
                !iterators_datapar_compatible<InIter2, OutIter>::value ||
                !iterator_datapar_compatible<InIter1>::value ||
                !iterator_datapar_compatible<InIter2>::value ||
                !iterator_datapar_compatible<OutIter>::value,
            hpx::util::tuple<InIter1, InIter2, OutIter>
        >::type
        call(InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
            OutIter dest, F && f)
        {
            return util::transform_binary_loop(parallel::v1::seq,
                first1, last1, first2, last2, dest, std::forward<F>(f));
        }
    };
}}}}

#endif
#endif

