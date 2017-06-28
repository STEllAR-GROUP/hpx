//  Copyright (c) 2017 Christopher Taylor
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#ifndef __HPXVALARRAY_H__
#define __HPXVALARRAY_H__

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>
#include <hpx/include/components.hpp>

#include <cstddef>
#include <valarray>

namespace hpx { namespace serialization
{
    template<typename T>
    void serialize(hpx::serialization::input_archive &ar,
        std::valarray<T> &arr, int /* version */) 
    {

        std::size_t sz = 0;
        ar & sz;
        arr.resize(sz);

        if(sz < 1) return;

        for(std::size_t i = 0; i < sz; ++i)
           ar >> arr[i];
    }

    template<typename T>
    void serialize(hpx::serialization::output_archive &ar,
        const std::valarray<T> arr, int /* version */)
    {

        const std::size_t sz = arr.size();
        ar & sz;
        for(auto v : arr)
          ar << v;
    }

}}

#endif

