//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2014-2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/intrusive_ptr.hpp
// hpxinspect:nodeprecatedname:boost::intrusive_ptr

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#if defined(HPX_SERIALIZATION_HAVE_BOOST_TYPES)
#include <hpx/serialization/detail/pointer.hpp>

#include <boost/intrusive_ptr.hpp>

namespace hpx { namespace serialization {

    template <typename T>
    void load(input_archive& ar, boost::intrusive_ptr<T>& ptr, unsigned)
    {
        detail::serialize_pointer_tracked(ar, ptr);
    }

    template <typename T>
    void save(output_archive& ar, boost::intrusive_ptr<T> const& ptr, unsigned)
    {
        detail::serialize_pointer_tracked(ar, ptr);
    }

    HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE(
        (template <typename T>), (boost::intrusive_ptr<T>) );
}}    // namespace hpx::serialization

#endif
