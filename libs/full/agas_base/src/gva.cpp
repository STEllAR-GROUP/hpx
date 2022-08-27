//  Copyright (c) 2020-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/agas_base/gva.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/util/ios_flags_saver.hpp>

#include <cstddef>
#include <ostream>

namespace hpx::agas {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void gva::save(Archive& ar, const unsigned int /*version*/) const
    {
        std::size_t lva = reinterpret_cast<std::size_t>(lva_);
        ar << prefix << type << count << lva << offset;
    }    //-V128

    template <typename Archive>
    void gva::load(Archive& ar, const unsigned int version)
    {
        if (version > HPX_AGAS_VERSION)
        {
            HPX_THROW_EXCEPTION(version_too_new, "gva::load",
                "trying to load GVA with unknown version");
        }

        std::size_t lva;
        ar >> prefix >> type >> count >> lva >> offset;    //-V128
        lva_ = reinterpret_cast<lva_type>(lva);
    }

    template void gva::save(
        hpx::serialization::output_archive& ar, const unsigned int) const;

    template void gva::load(
        hpx::serialization::input_archive& ar, const unsigned int);

    ///////////////////////////////////////////////////////////////////////////
    std::ostream& operator<<(std::ostream& os, gva const& addr)
    {
        hpx::util::ios_flags_saver ifs(os);
        os << "(" << addr.prefix << " "
           << components::get_component_type_name(addr.type) << " "
           << addr.count << " " << std::showbase << std::hex << addr.lva()
           << " " << addr.offset << ")";
        return os;
    }
}    // namespace hpx::agas
