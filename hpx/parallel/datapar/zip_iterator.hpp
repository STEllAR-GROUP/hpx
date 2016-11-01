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
#include <iterator>
#include <memory>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ... Iter>
    struct data_alignment_impl<hpx::util::zip_iterator<Iter...> >
    {
        template <std::size_t ... Is>
        static HPX_FORCEINLINE std::size_t
        call(hpx::util::zip_iterator<Iter...> const& it,
            hpx::util::detail::pack_c<std::size_t, Is...>)
        {
            auto const& t = it.get_iterator_tuple();
            std::size_t const sequencer[] = {
                std::size_t(0),
                (data_alignment(hpx::util::get<Is>(t)), std::size_t(0))...
            };
            return std::accumulate(
                &sequencer[0], &sequencer[sizeof(sequencer)/sizeof(sequencer[0])],
                std::size_t(0));
        }

        static HPX_FORCEINLINE std::size_t
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
    template <typename ... Iter, typename V>
    struct vector_pack_load<hpx::util::zip_iterator<Iter...>, V,
        typename std::enable_if<
            hpx::util::detail::all_of<
                parallel::util::detail::iterator_datapar_compatible<Iter> ...
            >::value
        >::type>
    {
        typedef hpx::util::tuple<
                typename rebind_pack<
                    V, typename std::iterator_traits<Iter>::value_type
                >::type ...
            > vector_pack_type;

        template <typename ... Iter_, std::size_t ... Is>
        static vector_pack_type aligned_pack(
            hpx::util::zip_iterator<Iter_...> const& iter,
            hpx::util::detail::pack_c<std::size_t, Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            return hpx::util::make_tuple(
                vector_pack_load<
                    Iter_, typename std::iterator_traits<Iter_>::value_type
                >::aligned(hpx::util::get<Is>(t)) ...
            );
        }

        template <typename ... Iter_>
        static vector_pack_type aligned(
            hpx::util::zip_iterator<Iter_...> const& iter)
        {
            return aligned_pack(iter,
                typename hpx::util::detail::make_index_pack<
                        sizeof...(Iter_)
                    >::type());
        }

        template <typename ... Iter_, std::size_t ... Is>
        static vector_pack_type unaligned_pack(
            hpx::util::zip_iterator<Iter_...> const& iter,
            hpx::util::detail::pack_c<std::size_t, Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            return hpx::util::make_tuple(
                vector_pack_load<
                    Iter_, typename std::iterator_traits<Iter_>::value_type
                >::unaligned(hpx::util::get<Is>(t)) ...
            );
        }

        template <typename ... Iter_>
        static vector_pack_type unaligned(
            hpx::util::zip_iterator<Iter_...> const& iter)
        {
            return unaligned_pack(iter,
                typename hpx::util::detail::make_index_pack<
                        sizeof...(Iter_)
                    >::type());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ... Iter>
    struct vector_pack_store<hpx::util::zip_iterator<Iter...>,
        typename std::enable_if<
            hpx::util::detail::all_of<
                parallel::util::detail::iterator_datapar_compatible<Iter> ...
            >::value
        >::type>
    {
        template <typename V_, typename ... Iter_, std::size_t ... Is>
        static void aligned_pack(
            V_ const& value, hpx::util::zip_iterator<Iter_...> const& iter,
            hpx::util::detail::pack_c<std::size_t, Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            int const sequencer[] = {
                0, (
                    vector_pack_store<Iter_>::aligned(
                        hpx::util::get<Is>(value), hpx::util::get<Is>(t)
                    ), 0
                ) ...
            };
            (void) sequencer;
        }

        template <typename V_, typename ... Iter_>
        static void aligned(V_ const& value,
            hpx::util::zip_iterator<Iter_...> const& iter)
        {
            aligned_pack(value, iter,
                typename hpx::util::detail::make_index_pack<
                        sizeof...(Iter_)
                    >::type());
        }

        template <typename V_, typename ... Iter_, std::size_t ... Is>
        static void unaligned_pack(
            V_ const& value, hpx::util::zip_iterator<Iter_...> const& iter,
            hpx::util::detail::pack_c<std::size_t, Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            int const sequencer[] = {
                0, (
                    vector_pack_store<Iter_>::unaligned(
                        hpx::util::get<Is>(value), hpx::util::get<Is>(t)
                    ), 0
                ) ...
            };
            (void) sequencer;
        }

        template <typename V_, typename ... Iter_>
        static void unaligned(V_ const& value,
            hpx::util::zip_iterator<Iter_...> const& iter)
        {
            unaligned_pack(value, iter,
                typename hpx::util::detail::make_index_pack<
                        sizeof...(Iter_)
                    >::type());
        }
    };
}}}

#endif
#endif
