//  Copyright (c) 2014 Anuj R. Sharma
//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http:// ww.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/components/containers/partitioned_vector/export_definitions.hpp>

#include <vector>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Data = std::vector<T>>
    class partitioned_vector;

    template <typename T, typename Data> class local_vector_iterator;
    template <typename T, typename Data> class const_local_vector_iterator;

    template <typename T, typename Data, typename BaseIter>
    class local_raw_vector_iterator;
    template <typename T, typename Data, typename BaseIter>
    class const_local_raw_vector_iterator;

    template <typename T, typename Data> class vector_iterator;
    template <typename T, typename Data> class const_vector_iterator;

    template <typename T, typename Data, typename BaseIter>
    class segment_vector_iterator;
    template <typename T, typename Data, typename BaseIter>
    class const_segment_vector_iterator;

    template <typename T, typename Data, typename BaseIter>
    class local_segment_vector_iterator;

    namespace server
    {
        template <typename T, typename Data = std::vector<T>>
        class partitioned_vector;

        template <typename T, typename Data = std::vector<T>>
        class partitioned_vector_partition;
    }
}

