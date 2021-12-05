//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/util.hpp>

#include <hpx/parcelport_lci/locality.hpp>

namespace hpx::parcelset::policies::lci {

    void locality::save(serialization::output_archive& ar) const
    {
        ar << rank_;
    }

    void locality::load(serialization::input_archive& ar)
    {
        ar >> rank_;
    }

    std::ostream& operator<<(std::ostream& os, locality const& loc) noexcept
    {
        hpx::util::ios_flags_saver ifs(os);
        os << loc.rank_;
        return os;
    }
}    // namespace hpx::parcelset::policies::lci

#endif
