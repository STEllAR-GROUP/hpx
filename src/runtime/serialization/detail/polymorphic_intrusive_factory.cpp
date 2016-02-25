//  Copyright (c) 2014 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/detail/polymorphic_intrusive_factory.hpp>

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/static.hpp>

#include <string>

namespace hpx { namespace serialization { namespace detail
{
    polymorphic_intrusive_factory& polymorphic_intrusive_factory::instance()
    {
        hpx::util::static_<polymorphic_intrusive_factory> factory;
        return factory.get();
    }

    void polymorphic_intrusive_factory::register_class(
        std::string const& name, ctor_type fun)
    {
        if (name.empty())
        {
            HPX_THROW_EXCEPTION(serialization_error
                , "polymorphic_intrusive_factory::register_class"
                , "Cannot register a factory with an empty name");
        }

        auto it = map_.find(name);
        if (it == map_.end())
        {
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 408000
            map_.emplace(name, fun);
#else
            map_.insert(ctor_map_type::value_type(name, fun));
#endif
        }
    }

    void* polymorphic_intrusive_factory::create(
        std::string const& name) const
    {
        return map_.at(name)();
    }
}}}
