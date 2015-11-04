//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/detail/polymorphic_id_factory.hpp>

namespace hpx { namespace serialization { namespace detail
{
    id_registry& id_registry::instance()
    {
        util::static_<id_registry> inst;
        return inst.get();
    }

    polymorphic_id_factory& polymorphic_id_factory::instance()
    {
        hpx::util::static_<polymorphic_id_factory> factory;
        return factory.get();
    }
}}}
