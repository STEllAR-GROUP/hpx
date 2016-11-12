//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERIALIZE_DATAPAR_BOOST_SIMD_NOV_10_2016_0338PM)
#define HPX_SERIALIZE_DATAPAR_BOOST_SIMD_NOV_10_2016_0338PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_BOOST_SIMD)
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/array.hpp>

#include <cstddef>

#include <boost/simd.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization
{
    template <typename T, std::size_t N, typename Abi>
    void serialize(input_archive & ar, boost::simd::pack<T, N, Abi> & v,
        unsigned)
    {
        ar & make_array(&v.storage(), v.size() * sizeof(T));
    }

    template <typename T, std::size_t N, typename Abi>
    void serialize(output_archive & ar, boost::simd::pack<T, N, Abi> const& v,
        unsigned)
    {
        ar & make_array(&v.storage(), v.size() * sizeof(T));
    }
}}

#endif
#endif

