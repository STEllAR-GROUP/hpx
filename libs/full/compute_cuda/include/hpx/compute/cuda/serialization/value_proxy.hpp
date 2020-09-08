//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__) && !defined(__CUDA_ARCH__)

#include <hpx/compute/cuda/value_proxy.hpp>
#include <hpx/serialization/serialize.hpp>

namespace hpx { namespace serialization {
    template <typename T>
    void serialize(
        input_archive& ar, cuda::experimental::value_proxy<T>& v, unsigned)
    {
        T t;
        ar >> t;
        v = t;
    }

    template <typename T>
    void serialize(output_archive& ar,
        cuda::experimental::value_proxy<T> const& v, unsigned)
    {
        ar << T(v);
    }
}}    // namespace hpx::serialization

#endif
