//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_SERIALIZATION_CUDA_VALUE_PROXY_HPP
#define HPX_COMPUTE_SERIALIZATION_CUDA_VALUE_PROXY_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__) && !defined(__CUDA_ARCH__)

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/compute/cuda/value_proxy.hpp>

namespace hpx { namespace serialization
{
    template <typename T>
    void serialize(input_archive & ar, compute::cuda::value_proxy<T> & v,
        unsigned)
    {
        T t;
        ar >> t;
        v = t;
    }

    template <typename T>
    void serialize(output_archive & ar, compute::cuda::value_proxy<T> const& v,
        unsigned)
    {
        ar << T(v);
    }
}}

#endif
#endif
