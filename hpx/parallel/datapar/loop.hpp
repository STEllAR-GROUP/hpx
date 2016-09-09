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
#include <hpx/traits/is_iterator.hpp>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>
#include <type_traits>

#include <Vc/Vc>

namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct datapar_loop_n
        {
            typedef typename hpx::util::decay<Iterator>::type iterator_type;

            typedef Vc::Vector<typename iterator_type::value_type> V;
            typedef Vc::Scalar::Vector<typename iterator_type::value_type> V1;

            ///////////////////////////////////////////////////////////////
            template <typename InIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                std::is_arithmetic<typename InIter::value_type>::value &&
                    hpx::traits::is_random_access_iterator<InIter>::value,
                InIter
            >::type
            call(InIter first, std::size_t count, F && f)
            {
                std::size_t len = count;

                for (/* */; detail::data_alignment(first) && len != 0;
                     (void) --len, ++first)
                {
                    datapar_loop_step<InIter>::call1(f, first);
                }

                std::int64_t lenV = std::int64_t(count - (V::Size + 1));
                for (/* */; lenV > 0;
                        lenV -= V::Size, len -= V::Size, first += V::Size)
                {
                    datapar_loop_step<InIter>::callv(f, first);
                }

                for (/* */; len != 0; (void) --len, ++first)
                {
                    datapar_loop_step<InIter>::call1(f, first);
                }

                return std::move(first);
            }

            template <typename InIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                !std::is_arithmetic<typename InIter::value_type>::value ||
                    !hpx::traits::is_random_access_iterator<InIter>::value,
                InIter
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
                std::is_arithmetic<typename InIter::value_type>::value &&
                    hpx::traits::is_random_access_iterator<InIter>::value,
                InIter
            >::type
            call(InIter first, std::size_t count, CancelToken& tok, F && f)
            {
                std::size_t len = count;

                for (/* */; detail::data_alignment(first) && len != 0;
                     (void) --len, ++first)
                {
                    V1 tmp(std::addressof(*first), Vc::Aligned);
                    f(&tmp);
                    if (tok.was_cancelled())
                        return first;
                    tmp.store(std::addressof(*first), Vc::Aligned);
                }

                std::int64_t lenV = std::int64_t(count - (V::Size + 1));
                for (/* */; lenV > 0;
                        lenV -= V::Size, len -= V::Size, first += V::Size)
                {
                    V tmp(std::addressof(*first), Vc::Aligned);
                    f(&tmp);
                    if (tok.was_cancelled())
                        return first;
                    tmp.store(std::addressof(*first), Vc::Aligned);
                }

                for (/* */; len != 0; (void) --len, ++first)
                {
                    V1 tmp(std::addressof(*first), Vc::Aligned);
                    f(&tmp);
                    if (tok.was_cancelled())
                        return first;
                    tmp.store(std::addressof(*first), Vc::Aligned);
                }

                return first;
            }

            template <typename InIter, typename CancelToken, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                !std::is_arithmetic<typename InIter::value_type>::value ||
                    !hpx::traits::is_random_access_iterator<InIter>::value,
                InIter
            >::type
            call(InIter first, std::size_t count, CancelToken& tok, F && f)
            {
                return util::loop_n(parallel::v1::seq, first, count, tok,
                    std::forward<F>(f));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Abi>
        inline std::size_t count_bits(Vc::Mask<T, Abi> const& mask)
        {
            return mask.count();
        }
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
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function starting from a given
        // iterator position.
        template <typename Iterator>
        struct datapar_loop
        {
            typedef typename hpx::util::decay<Iterator>::type iterator_type;
            typedef Vc::Vector<typename iterator_type::value_type> V;

            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
                std::is_arithmetic<typename Begin::value_type>::value &&
                    hpx::traits::is_random_access_iterator<Begin>::value &&
                    hpx::traits::is_random_access_iterator<End>::value,
                Begin
            >::type
            call(Begin first, End last, F && f)
            {
                for (/* */; data_alignment(first) && first != last; ++first)
                {
                    datapar_loop_step<Begin>::call1(f, first);
                }

                for (End const lastV = last - (V::Size + 1);
                    first < lastV; first += V::Size)
                {
                    datapar_loop_step<Begin>::callv(f, first);
                }

                for (/* */; first != last; ++first)
                {
                    datapar_loop_step<Begin>::call1(f, first);
                }

                return first;
            }

            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static typename std::enable_if<
               !std::is_arithmetic<typename Begin::value_type>::value ||
                   !hpx::traits::is_random_access_iterator<Begin>::value ||
                   !hpx::traits::is_random_access_iterator<End>::value,
                Begin
            >::type
            call(Begin first, End last, F && f)
            {
                return util::loop(parallel::v1::seq, first, last,
                    std::forward<F>(f));
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin
    loop(parallel::v1::datapar_execution_policy, Begin begin, End end, F && f)
    {
        return detail::datapar_loop<Begin>::call(begin, end, std::forward<F>(f));
    }

    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin
    loop(parallel::v1::datapar_task_execution_policy, Begin begin, End end,
        F && f)
    {
        return detail::datapar_loop<Begin>::call(begin, end, std::forward<F>(f));
    }
}}}

#endif
#endif

