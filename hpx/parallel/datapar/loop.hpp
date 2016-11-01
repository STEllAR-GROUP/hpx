//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_UTIL_LOOP_SEP_07_2016_1217PM)
#define HPX_PARALLEL_DATAPAR_UTIL_LOOP_SEP_07_2016_1217PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/datapar/execution_policy_fwd.hpp>
#include <hpx/parallel/datapar/iterator_helpers.hpp>
#include <hpx/parallel/traits/vector_pack_alignment_size.hpp>
#include <hpx/parallel/traits/vector_pack_load_store.hpp>
#include <hpx/parallel/traits/vector_pack_type.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/util/decay.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Vector>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        typename Vector::value_type
        extract_value(parallel::v1::datapar_execution_policy,
            Vector const& value)
        {
            static_assert(traits::vector_pack_is_scalar<Vector>::value,
                "this should be called with a scalar only");
            return value[0];
        }

        template <typename Vector>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        typename Vector::value_type
        extract_value(parallel::v1::datapar_task_execution_policy,
            Vector const& value)
        {
            static_assert(traits::vector_pack_is_scalar<Vector>::value,
                "this should be called with a scalar only");
            return value[0];
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Vector>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        typename traits::vector_pack_type<
            typename hpx::util::decay<Vector>::type::value_type, 1
        >::type
        accumulate_values(parallel::v1::datapar_execution_policy,
            F && f, Vector const& value)
        {
            typedef typename hpx::util::decay<Vector>::type vector_type;
            typedef typename vector_type::value_type entry_type;

            entry_type accum = value[0];
            for(size_t i = 1; i != value.size(); ++i)
            {
                accum = f(accum, entry_type(value[i]));
            }

            return typename traits::vector_pack_type<entry_type, 1>::type(accum);
        }

        template <typename F, typename Vector>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        typename traits::vector_pack_type<
            typename hpx::util::decay<Vector>::type::value_type, 1
        >::type
        accumulate_values(parallel::v1::datapar_task_execution_policy,
            F && f, Vector const& value)
        {
            typedef typename hpx::util::decay<Vector>::type vector_type;
            typedef typename vector_type::value_type entry_type;

            entry_type accum = value[0];
            for(size_t i = 1; i != value.size(); ++i)
            {
                accum = f(accum, entry_type(value[i]));
            }

            return typename traits::vector_pack_type<entry_type, 1>::type(accum);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Vector, typename T>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        typename traits::vector_pack_type<T, 1>::type
        accumulate_values(parallel::v1::datapar_execution_policy,
            F && f, Vector const& value, T accum)
        {
            for(size_t i = 0; i != value.size(); ++i)
            {
                accum = f(accum, T(value[i]));
            }

            return typename traits::vector_pack_type<T, 1>::type(accum);
        }

        template <typename F, typename Vector, typename T>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        typename traits::vector_pack_type<T, 1>::type
        accumulate_values(parallel::v1::datapar_task_execution_policy,
            F && f, Vector const& value, T accum)
        {
            for(size_t i = 0; i != value.size(); ++i)
            {
                accum = f(accum, T(value[i]));
            }

            return typename traits::vector_pack_type<T, 1>::type(accum);
        }

        ///////////////////////////////////////////////////////////////////////
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
            template <typename Iter>
            static bool call(Iter const& first, Iter const& last)
            {
                typedef typename std::iterator_traits<Iter>::value_type
                    value_type;
                typedef typename traits::vector_pack_type<value_type>::type V;

                return traits::vector_pack_size<Iter, V>::value <=
                        std::distance(first, last);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function starting from a given
        // iterator position.
        template <typename Iterator>
        struct datapar_loop
        {
            typedef typename hpx::util::decay<Iterator>::type iterator_type;
            typedef typename std::iterator_traits<iterator_type>::value_type
                value_type;

            typedef typename traits::vector_pack_type<value_type> V;

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

                static std::size_t const size =
                    traits::vector_pack_size<Begin, V>::value;

                End const lastV = last - (size + 1);
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
                while (it != end)
                {
                    datapar_loop_step<Begin>::call1(f, it);
                }
                return it;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function starting from a given
        // iterator position.
        template <typename VecOnly, typename Iter1, typename Iter2>
        struct datapar_loop2;

        template <typename Iter1, typename Iter2>
        struct datapar_loop2<std::true_type, Iter1, Iter2>
        {
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

                typedef typename traits::vector_pack_type<value_type>::type V;

                if (detail::data_alignment(it1) || detail::data_alignment(it2))
                {
                    return std::make_pair(std::move(it1), std::move(it2));
                }

                static std::size_t const size =
                    traits::vector_pack_size<InIter1, V>::value;

                InIter1 const last1V = last1 - (size + 1);
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

        template <typename Iter1, typename Iter2>
        struct datapar_loop2<std::false_type, Iter1, Iter2>
        {
            template <typename InIter1, typename InIter2, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            static std::pair<InIter1, InIter2>
            call(InIter1 it1, InIter1 last1, InIter2 it2, F && f)
            {
                return util::detail::loop2<InIter1, InIter2>::call(
                    it1, last1, it2, std::forward<F>(f));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct datapar_loop_n
        {
            typedef typename hpx::util::decay<Iterator>::type iterator_type;
            typedef typename std::iterator_traits<iterator_type>::value_type
                value_type;

            typedef typename traits::vector_pack_type<value_type>::type V;

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

                static std::size_t const size =
                    traits::vector_pack_size<InIter, V>::value;

                for (std::int64_t lenV = std::int64_t(count - (size + 1));
                        lenV > 0; lenV -= size, len -= size)
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
                for (/* */; count != 0; --count)
                {
                    datapar_loop_step<InIter>::call1(f, first);
                }
                return first;
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Iter1, typename Iter2>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    auto loop_step(parallel::v1::datapar_execution_policy, std::false_type,
            F && f, Iter1& it1, Iter2& it2)
    ->  decltype(
            detail::datapar_loop_step2<Iter1, Iter2>::call1(
                std::forward<F>(f), it1, it2)
            )
    {
        return detail::datapar_loop_step2<Iter1, Iter2>::call1(
            std::forward<F>(f), it1, it2);
    }

    template <typename F, typename Iter1, typename Iter2>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    auto loop_step(parallel::v1::datapar_execution_policy, std::true_type,
            F && f, Iter1& it1, Iter2& it2)
    ->  decltype(
            detail::datapar_loop_step2<Iter1, Iter2>::callv(
                std::forward<F>(f), it1, it2)
            )
    {
        return detail::datapar_loop_step2<Iter1, Iter2>::callv(
            std::forward<F>(f), it1, it2);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename Iter1, typename Iter2>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    auto loop_step(parallel::v1::datapar_task_execution_policy, std::false_type,
            F && f, Iter1& it1, Iter2& it2)
    ->  decltype(
            detail::datapar_loop_step2<Iter1, Iter2>::call1(
                std::forward<F>(f), it1, it2)
            )
    {
        return detail::datapar_loop_step2<Iter1, Iter2>::call1(
            std::forward<F>(f), it1, it2);
    }

    template <typename F, typename Iter1, typename Iter2>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    auto loop_step(parallel::v1::datapar_task_execution_policy, std::true_type,
            F && f, Iter1& it1, Iter2& it2)
    ->  decltype(
            detail::datapar_loop_step2<Iter1, Iter2>::callv(
                std::forward<F>(f), it1, it2)
            )
    {
        return detail::datapar_loop_step2<Iter1, Iter2>::callv(
            std::forward<F>(f), it1, it2);
    }

    ///////////////////////////////////////////////////////////////////////////
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
    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin
    loop(parallel::v1::datapar_execution_policy, Begin begin, End end, F && f)
    {
        return detail::datapar_loop<Begin>::call(begin, end, std::forward<F>(f));
    }

    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin
    loop(parallel::v1::datapar_task_execution_policy, Begin begin, End end, F && f)
    {
        return detail::datapar_loop<Begin>::call(begin, end, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename VecOnly, typename Iter1, typename Iter2, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::pair<Iter1, Iter2>
    loop2(parallel::v1::datapar_execution_policy, VecOnly,
        Iter1 first1, Iter1 last1, Iter2 first2, F && f)
    {
        return detail::datapar_loop2<VecOnly, Iter1, Iter2>::call(
            first1, last1, first2, std::forward<F>(f));
    }

    template <typename VecOnly, typename Iter1, typename Iter2, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE std::pair<Iter1, Iter2>
    loop2(parallel::v1::datapar_task_execution_policy, VecOnly,
        Iter1 first1, Iter1 last1, Iter2 first2, F && f)
    {
        return detail::datapar_loop2<VecOnly, Iter1, Iter2>::call(
            first1, last1, first2, std::forward<F>(f));
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
}}}

#endif
#endif

