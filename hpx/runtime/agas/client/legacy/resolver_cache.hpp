////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_63FAE67B_EB85_4A1A_B612_F98559D9F924)
#define HPX_63FAE67B_EB85_4A1A_B612_F98559D9F924

#include <map>

#include <boost/cache/local_cache.hpp>
#include <boost/cache/entries/lfu_entry.hpp>

#include <hpx/lcos/mutex.hpp>
#include <hpx/lcos/mutex.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/agas/network/gva.hpp>
#include <hpx/util/runtime_configuration.hpp>

namespace hpx { namespace agas { namespace legacy
{

template <typename Protocol>
struct resolver_cache
{
  protected:
    typedef gva<Protocol> gva_type;
    typedef typename gva_type::count_type count_type;

    struct cache_key
    { // {{{ cache_key implementation
        cache_key()
          : id(), count(0)
        {}

        explicit cache_key(naming::gid_type const& id_,
                           count_type count_ = 1)
          : id(id_), count(count_)
        {}

        naming::gid_type id;
        count_type count;

        friend bool operator<(cache_key const& lhs, cache_key const& rhs)
        { return (lhs.id + (lhs.count - 1)) < rhs.id; }

        friend bool operator==(cache_key const& lhs, cache_key const& rhs)
        { return (lhs.id == rhs.id) && (lhs.count == rhs.count); }
    }; // }}}
    
    struct erase_policy
    { // {{{ erase_policy implementation
        erase_policy(naming::gid_type const& id, count_type count)
          : entry(id, count)
        {}

        typedef std::pair<
            cache_key, boost::cache::entries::lfu_entry<gva_type>
        > entry_type;

        bool operator()(entry_type const& p) const
        { return p.first == entry; }

        cache_key entry;
    }; // }}}

    typedef boost::cache::entries::lfu_entry<gva_type> entry_type;

    typedef hpx::lcos::mutex cache_mutex_type;

    typedef boost::cache::local_cache<
        cache_key, entry_type, 
        std::less<entry_type>, boost::cache::policies::always<entry_type>,
        std::map<cache_key, entry_type>
    > cache_type;

    cache_mutex_type cache_mtx_;
    cache_type gva_cache_;

    resolver_cache(util::runtime_configuration const& ini_)
        : gva_cache_(ini_.get_agas_cache_size())
    {
        if (HPX_UNLIKELY(ini_.get_agas_cache_size() < 3))
        {
            HPX_THROW_IN_CURRENT_FUNC(bad_parameter, 
                "AGAS cache size must be at least 3");
        }
    }  
};

}}}

#endif // HPX_63FAE67B_EB85_4A1A_B612_F98559D9F924

