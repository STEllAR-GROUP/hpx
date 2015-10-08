//  Copyright (c) 2014 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/detail/polymorphic_intrusive_factory.hpp>

namespace hpx { namespace serialization { namespace detail
{
    polymorphic_intrusive_factory& polymorphic_intrusive_factory::instance()
    {
        hpx::util::static_<polymorphic_intrusive_factory> factory;
        return factory.get();
    }
}}}

