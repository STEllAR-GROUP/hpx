//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file action_factory.cpp

#include <hpx/config.hpp>
#include <hpx/runtime/actions/action_factory.hpp>
#include <hpx/runtime/actions/action_support.hpp>

#include <hpx/util/static.hpp>
#include <hpx/util/jenkins_hash.hpp>

#include <boost/foreach.hpp>

namespace hpx { namespace actions
{
    action_factory::ctor_map::const_iterator action_factory::locate(
        boost::uint32_t hash, std::string const& name) const
    {
        typedef std::pair<
            ctor_map::const_iterator, ctor_map::const_iterator
        > equal_range_type;

        equal_range_type r = ctor_map_.equal_range(hash);
        if (r.first != r.second)
        {
            ctor_map::const_iterator it = r.first;
            if (++it == r.second)
            {
                // there is only one match in the map
                return r.first;
            }

            // there is more than one entry with the same hash in the map
            for (it = r.first; it != r.second; ++it)
            {
                if ((*it).second.first == name)
                    return it;
            }

            // fall through...
        }
        return ctor_map_.end();
    }

    boost::shared_ptr<base_action> action_factory::create(
        std::string const & name)
    {
        action_factory const & factory = action_factory::get_instance();
        ctor_map::const_iterator it = factory.locate(
            util::jenkins_hash()(name), name);

        if (it != factory.ctor_map_.end())
            return ((*it).second.second)();

        std::string error = "Can not find action '";
        error += name;
        error += "' in type registry";
        HPX_THROW_EXCEPTION(bad_action_code
            , "action_factory::create"
            , error);
        return boost::shared_ptr<base_action>();
    }

    void action_factory::add_action(std::string const & name, ctor_type ctor)
    {
        boost::uint32_t hash = util::jenkins_hash()(name);
        ctor_map::const_iterator it = locate(hash, name);
        if (it != ctor_map_.end())
            return;

        ctor_map_.insert(std::make_pair(hash, std::make_pair(name, ctor)));
    }

    HPX_EXPORT action_factory & action_factory::get_instance()
    {
        util::static_<action_factory> factory;
        return factory.get();
    }
}}
