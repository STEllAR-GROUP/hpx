//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/format.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/serialization/serialize.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::naming {

    template <typename Archive>
    void address::save(Archive& ar, unsigned int /* version */) const
    {
        // clang-format off
        std::size_t address = reinterpret_cast<std::size_t>(address_);
        ar & locality_ & type_ & address;
        // clang-format on
    }

    template <typename Archive>
    void address::load(Archive& ar, unsigned int /* version */)
    {
        // clang-format off
        std::size_t address;
        ar & locality_ & type_ & address;
        address_ = reinterpret_cast<address_type>(address);
        // clang-format on
    }

    template HPX_EXPORT void address::save(
        serialization::output_archive&, unsigned int) const;

    template HPX_EXPORT void address::load(
        serialization::input_archive&, unsigned int);
}    // namespace hpx::naming
