//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_BOOST_SIMD_ITERATOR_HELPERS_SEP_22_2016_0228PM)
#define HPX_PARALLEL_DATAPAR_BOOST_SIMD_ITERATOR_HELPERS_SEP_22_2016_0228PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_BOOST_SIMD)
#include <hpx/util/tuple.hpp>
#include <hpx/util/zip_iterator.hpp>

#include <cstddef>
#include <iterator>

#include <boost/simd.hpp>

namespace hpx { namespace parallel { namespace traits { namespace detail
{
    template <typename ... Iter, typename T, std::size_t N, typename Abi>
    struct vector_pack_size<
        hpx::util::zip_iterator<Iter...>, boost::simd::pack<T, N, Abi> >
    {
        typedef boost::simd::pack<
                typename hpx::util::detail::at_index<
                    0, typename std::iterator_traits<Iter>::value_type...
                >::type
            > rebound_pack_type;

        static std::size_t const value = rebound_pack_type::static_size;
    };
}}}}

#endif
#endif
