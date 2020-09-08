//  Copyright (c) 2015-2020 Hartmut Kaiser
//  Copyright (c) 2015-2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/detail/preprocess_gid_types.hpp>
#include <hpx/serialization/detail/extra_archive_data.hpp>

namespace hpx { namespace serialization { namespace detail {

    // This is explicitly instantiated to ensure that the id is stable across
    // shared libraries. MSVC and gcc/clang require different handling of
    // exported explicitly instantiated templates.
#if defined(HPX_MSVC)
    template struct HPX_EXPORT
        extra_archive_data_id_helper<preprocess_gid_types>;
#else
    template struct extra_archive_data_id_helper<preprocess_gid_types>;
#endif
}}}    // namespace hpx::serialization::detail
