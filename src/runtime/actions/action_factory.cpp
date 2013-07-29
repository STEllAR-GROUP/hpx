//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file action_factory.cpp

#include <hpx/config.hpp>
#include <hpx/runtime/actions/action_factory.hpp>
#include <hpx/runtime/actions/action_support.hpp>

#include <hpx/util/static.hpp>

#include <boost/foreach.hpp>

namespace hpx { namespace actions {
    boost::shared_ptr<base_action> action_factory::create(std::string const & name)
    {
        action_factory const & factory = action_factory::get_instance();

        ctor_map::const_iterator it = factory.ctor_map_.find(name);
        if(it == factory.ctor_map_.end())
        {
            std::string error = "Can not find action ";
            error += name;
            error += " in map";
            HPX_THROW_EXCEPTION(bad_action_code,
                "action_factory::create"
              , error);
        }

        return (it->second)();
    }

    void action_factory::add_action(std::string const & name, ctor_type ctor)
    {
        ctor_map::iterator it = ctor_map_.find(name);
        if(it != ctor_map_.end())
        {
            return;
        }

        ctor_map_.insert(std::make_pair(name, ctor));
    }

    HPX_ALWAYS_EXPORT action_factory & action_factory::get_instance()
    {
        util::static_<action_factory> factory;
        return factory.get();
    }
}}
