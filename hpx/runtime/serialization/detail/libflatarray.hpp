//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERIALIZE_DATAPAR_LIBFLATARRAY_NOV_10_2016_0355PM)
#define HPX_SERIALIZE_DATAPAR_LIBFLATARRAY_NOV_10_2016_0355PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR_LIBFLATARRAY)
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/array.hpp>

#include <array>
#include <cstddef>

#include <libflatarray/flat_array.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization
{
    template <typename T, std::size_t N>
    void serialize(input_archive & ar, LibFlatArray::short_vec<T, N> & v,
        unsigned)
    {
        std::array<T, N> data;
        ar & data;
        v.load(data.data());
    }

    template <typename T, std::size_t N>
    void serialize(output_archive & ar, LibFlatArray::short_vec<T, N> const& v,
        unsigned)
    {
        std::array<T, N> data;
        v.store(data.data());
        ar & data;
    }
}}

#endif
#endif

