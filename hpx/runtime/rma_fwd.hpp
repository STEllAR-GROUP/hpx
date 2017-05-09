//  Copyright (c) 2017 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_RMA_FWD_HPP
#define HPX_RUNTIME_RMA_FWD_HPP

#include <hpx/config.hpp>
//
#include <vector>

namespace hpx { namespace  parcelset { namespace rma
{
    class memory_region;

    template <typename T>
    struct allocator;

    template <typename T>
    using rma_vector = std::vector<T, hpx::parcelset::rma::allocator<T>>;
}}}

namespace hpx { namespace serialization
{
    namespace detail
    {
        template <typename T>
        void save_impl(output_archive &,
            const hpx::parcelset::rma::rma_vector<T> & , std::true_type);

        template <typename T>
        void load_impl(input_archive &,
            hpx::parcelset::rma::rma_vector<T> & , std::true_type);
    }
}}

#endif
