//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2016 Matthias Kretz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_VC_ITERATOR_HELPERS_SEP_09_2016_0143PM)
#define HPX_PARALLEL_DATAPAR_VC_ITERATOR_HELPERS_SEP_09_2016_0143PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_VC)
#include <hpx/util/tuple.hpp>
#include <hpx/util/zip_iterator.hpp>

#include <cstddef>
#include <iterator>

#include <Vc/Vc>

namespace hpx { namespace parallel { namespace traits { namespace detail
{
    template <typename ... Iter, typename T, typename Abi>
    struct vector_pack_size<
        hpx::util::zip_iterator<Iter...>, Vc::Vector<T, Abi> >
    {
        typedef Vc::Vector<
                typename hpx::util::detail::at_index<
                    0, typename std::iterator_traits<Iter>::value_type...
                >::type
            > rebound_pack_type;

        static std::size_t const value = rebound_pack_type::Size;
    };

    template <typename ... Iter, typename T, typename Abi>
    struct vector_pack_size<
        hpx::util::zip_iterator<Iter...>, Vc::Scalar::Vector<T, Abi> >
    {
        typedef Vc::Vector<
                typename hpx::util::detail::at_index<
                    0, typename std::iterator_traits<Iter>::value_type...
                >::type
            > rebound_pack_type;

        static std::size_t const value = rebound_pack_type::Size;
    };
}}}}

#endif
#endif

