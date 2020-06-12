//  Copyright (c) 2017 Christopher Taylor
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_decl.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/modules/errors.hpp>

#include <string>

namespace hpx { namespace serialization
{
    template <typename T>
    void serialize(input_archive& ar, hpx::partitioned_vector<T>& v, unsigned)
    {
        std::string pvec_registered_name;
        ar >> pvec_registered_name;
        v.connect_to(pvec_registered_name).get();
    }

    template <typename T>
    void serialize(
        output_archive& ar, const hpx::partitioned_vector<T>& v, unsigned)
    {
        std::string pvec_registered_name = v.registered_name();
        if (pvec_registered_name.empty())
        {
            HPX_THROW_EXCEPTION(
                hpx::invalid_status,
                "hpx::serialization::serialize",
                "partitioned_vector is not registered");
        }
        ar << pvec_registered_name;
    }
}}

