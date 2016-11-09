//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_ZIP_ITERATOR_SEP_30_2016_1251PM)
#define HPX_PARALLEL_DATAPAR_ZIP_ITERATOR_SEP_30_2016_1251PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)

#include <hpx/util/detail/pack.hpp>
#include <hpx/util/zip_iterator.hpp>
#include <hpx/traits/is_iterator.hpp>

#include <hpx/parallel/datapar/iterator_helpers.hpp>
#include <hpx/parallel/traits/vector_pack_alignment_size.hpp>
#include <hpx/parallel/traits/vector_pack_load_store.hpp>
#include <hpx/parallel/traits/vector_pack_type.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <numeric>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ... Iter>
    struct is_data_aligned_impl<hpx::util::zip_iterator<Iter...> >
    {
        template <std::size_t ... Is>
        static HPX_FORCEINLINE bool
        call(hpx::util::zip_iterator<Iter...> const& it,
            hpx::util::detail::pack_c<std::size_t, Is...>)
        {
            auto const& t = it.get_iterator_tuple();
            bool const sequencer[] = {
                false, is_data_aligned(hpx::util::get<Is>(t)) ...
            };
            return std::any_of(
                &sequencer[1], &sequencer[sizeof...(Is) + 1],
                [](bool val) { return val; });
        }

        static HPX_FORCEINLINE bool
        call(hpx::util::zip_iterator<Iter...> const& it)
        {
            return call(it,
                typename hpx::util::detail::make_index_pack<
                        sizeof...(Iter)
                    >::type());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ... Iter>
    struct iterator_datapar_compatible_impl<hpx::util::zip_iterator<Iter...> >
      : hpx::util::detail::all_of<
            std::is_arithmetic<
                typename std::iterator_traits<Iter>::value_type
            > ...
        >
    {};
}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Tuple, typename ... Iter, std::size_t ... Is>
        Tuple aligned_pack(hpx::util::zip_iterator<Iter...> const& iter,
            hpx::util::detail::pack_c<std::size_t, Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            return hpx::util::make_tuple(
                vector_pack_load<
                    typename hpx::util::tuple_element<Is, Tuple>::type
                >::aligned(hpx::util::get<Is>(t)) ...
            );
        }

        template <typename Tuple, typename ... Iter, std::size_t ... Is>
        Tuple unaligned_pack(hpx::util::zip_iterator<Iter...> const& iter,
            hpx::util::detail::pack_c<std::size_t, Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            return hpx::util::make_tuple(
                vector_pack_load<
                    typename hpx::util::tuple_element<Is, Tuple>::type
                >::unaligned(hpx::util::get<Is>(t)) ...
            );
        }
    }

    template <typename ... Vector>
    struct vector_pack_load<hpx::util::tuple<Vector...> >
    {
        typedef hpx::util::tuple<Vector...> tuple_type;

        template <typename ... Iter>
        static tuple_type
        aligned(hpx::util::zip_iterator<Iter...> const& iter)
        {
            return traits::detail::aligned_pack<tuple_type>(iter,
                typename hpx::util::detail::make_index_pack<
                        sizeof...(Iter)
                    >::type());
        }

        template <typename ... Iter>
        static tuple_type
        unaligned(hpx::util::zip_iterator<Iter...> const& iter)
        {
            return traits::detail::unaligned_pack<tuple_type>(iter,
                typename hpx::util::detail::make_index_pack<
                        sizeof...(Iter)
                    >::type());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Tuple, typename ... Iter, std::size_t ... Is>
        void aligned_pack(
            Tuple const& value, hpx::util::zip_iterator<Iter...> const& iter,
            hpx::util::detail::pack_c<std::size_t, Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            int const sequencer[] = {
                0, (
                    vector_pack_store<
                        typename hpx::util::tuple_element<Is, Tuple>::type
                    >::aligned(
                        hpx::util::get<Is>(value), hpx::util::get<Is>(t)
                    ), 0
                ) ...
            };
            (void) sequencer;
        }

        template <typename Tuple, typename ... Iter, std::size_t ... Is>
        void unaligned_pack(
            Tuple const& value, hpx::util::zip_iterator<Iter...> const& iter,
            hpx::util::detail::pack_c<std::size_t, Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            int const sequencer[] = {
                0, (
                    vector_pack_store<
                        typename hpx::util::tuple_element<Is, Tuple>::type
                    >::unaligned(
                        hpx::util::get<Is>(value), hpx::util::get<Is>(t)
                    ), 0
                ) ...
            };
            (void) sequencer;
        }
    }

    template <typename ... Vector>
    struct vector_pack_store<hpx::util::tuple<Vector...> >
    {
        template <typename V, typename ... Iter>
        static void
        aligned(V const& value, hpx::util::zip_iterator<Iter...> const& iter)
        {
            traits::detail::aligned_pack(value, iter,
                typename hpx::util::detail::make_index_pack<
                        sizeof...(Iter)
                    >::type());
        }

        template <typename V, typename ... Iter>
        static void
        unaligned(V const& value, hpx::util::zip_iterator<Iter...> const& iter)
        {
            traits::detail::unaligned_pack(value, iter,
                typename hpx::util::detail::make_index_pack<
                        sizeof...(Iter)
                    >::type());
        }
    };
}}}

#endif
#endif
