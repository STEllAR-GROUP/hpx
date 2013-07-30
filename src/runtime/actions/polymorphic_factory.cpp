//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file polymorphic_factory.cpp

#include <hpx/config.hpp>
#include <hpx/runtime/actions/polymorphic_factory.hpp>
#include <hpx/runtime/actions/action_support.hpp>

#include <hpx/util/static.hpp>
#include <hpx/util/jenkins_hash.hpp>

#include <boost/foreach.hpp>

namespace hpx { namespace actions
{
    template <typename Base>
    typename polymorphic_factory<Base>::ctor_map::const_iterator
    polymorphic_factory<Base>::locate(boost::uint32_t hash,
            std::string const& name) const
    {
        typedef std::pair<
            typename ctor_map::const_iterator, typename ctor_map::const_iterator
        > equal_range_type;

        equal_range_type r = ctor_map_.equal_range(hash);
        if (r.first != r.second)
        {
            typename ctor_map::const_iterator it = r.first;
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

    template <typename Base>
    boost::shared_ptr<Base> polymorphic_factory<Base>::create(
        std::string const & name)
    {
        polymorphic_factory const & factory = polymorphic_factory::get_instance();
        typename ctor_map::const_iterator it = factory.locate(
            util::jenkins_hash()(name), name);

        if (it != factory.ctor_map_.end())
            return ((*it).second.second)();

        std::string error = "Can not find action '";
        error += name;
        error += "' in type registry";
        HPX_THROW_EXCEPTION(bad_action_code
            , "polymorphic_factory::create"
            , error);
        return boost::shared_ptr<Base>();
    }

    template <typename Base>
    void polymorphic_factory<Base>::add_factory_function(
        std::string const & name, ctor_type ctor)
    {
        boost::uint32_t hash = util::jenkins_hash()(name);
        typename ctor_map::const_iterator it = locate(hash, name);
        if (it != ctor_map_.end())
            return;

        ctor_map_.insert(std::make_pair(hash, std::make_pair(name, ctor)));
    }

    template <typename Base>
    polymorphic_factory<Base> & polymorphic_factory<Base>::get_instance()
    {
        util::static_<polymorphic_factory> factory;
        return factory.get();
    }

    template struct HPX_EXPORT polymorphic_factory<base_action>;
    template struct HPX_EXPORT polymorphic_factory<continuation>;
}}
