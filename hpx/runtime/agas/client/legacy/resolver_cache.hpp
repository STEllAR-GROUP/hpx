////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_63FAE67B_EB85_4A1A_B612_F98559D9F924)
#define HPX_63FAE67B_EB85_4A1A_B612_F98559D9F924

#include <boost/atomic.hpp>
#include <boost/cstdint.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/cache/local_cache.hpp>
#include <boost/cache/entries/lfu_entry.hpp>

#include <hpx/lcos/mutex.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/agas/network/gva.hpp>
#include <hpx/util/runtime_configuration.hpp>

namespace hpx { namespace agas { namespace legacy
{

template <typename Protocol>
struct resolver_cache
{
  protected:
    // {{{ types
    typedef gva<Protocol> gva_type;
    typedef typename gva_type::count_type count_type;
    typedef hpx::lcos::mutex cache_mutex_type;
    // }}}

    // {{{ gva cache
    struct gva_cache_key
    { // {{{ gva_cache_key implementation
        gva_cache_key()
          : id(), count(0)
        {}

        explicit gva_cache_key(naming::gid_type const& id_,
                               count_type count_ = 1)
          : id(id_), count(count_)
        {}

        naming::gid_type id;
        count_type count;

        friend bool
        operator<(gva_cache_key const& lhs, gva_cache_key const& rhs)
        { return (lhs.id + (lhs.count - 1)) < rhs.id; }

        friend bool
        operator==(gva_cache_key const& lhs, gva_cache_key const& rhs)
        { return (lhs.id == rhs.id) && (lhs.count == rhs.count); }
    }; // }}}
    
    struct gva_erase_policy
    { // {{{ gva_erase_policy implementation
        gva_erase_policy(naming::gid_type const& id, count_type count)
          : entry(id, count)
        {}

        typedef std::pair<
            gva_cache_key, boost::cache::entries::lfu_entry<gva_type>
        > entry_type;

        bool operator()(entry_type const& p) const
        { return p.first == entry; }

        gva_cache_key entry;
    }; // }}}

    typedef boost::cache::entries::lfu_entry<gva_type> gva_entry_type;

    typedef boost::cache::local_cache<
        gva_cache_key, gva_entry_type, 
        std::less<gva_entry_type>,
        boost::cache::policies::always<gva_entry_type>,
        std::map<gva_cache_key, gva_entry_type>
    > gva_cache_type;
    // }}}

    // {{{ locality cache 
    typedef boost::cache::entries::lfu_entry<naming::gid_type>
        locality_entry_type;

    typedef boost::cache::local_cache<naming::locality, locality_entry_type>
        locality_cache_type;
    // }}}

    typedef boost::atomic<boost::uint32_t> console_cache_type;

    cache_mutex_type gva_cache_mtx_;
    gva_cache_type gva_cache_;
    
    cache_mutex_type locality_cache_mtx_;
    locality_cache_type locality_cache_;

    console_cache_type console_cache_;

    resolver_cache(util::runtime_configuration const& ini_)
        : gva_cache_(ini_.get_agas_gva_cache_size())
        , locality_cache_(ini_.get_agas_locality_cache_size())
        , console_cache_(0)
    {
        if (HPX_UNLIKELY(ini_.get_agas_gva_cache_size() < 3))
        {
            HPX_THROW_IN_CURRENT_FUNC(bad_parameter, 
                "AGAS GVA cache size must be at least 3");
        }

        if (HPX_UNLIKELY(ini_.get_agas_locality_cache_size() < 1))
        {
            HPX_THROW_IN_CURRENT_FUNC(bad_parameter, 
                "AGAS locality cache size must be at least 1");
        }
    }  
};

}}}

#endif // HPX_63FAE67B_EB85_4A1A_B612_F98559D9F924

