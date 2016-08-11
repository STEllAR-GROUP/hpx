//  Copyright (c) 2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <list>

namespace hpx { namespace serialization
{
    void output_archive::add_gid(
        naming::gid_type const & gid,
        naming::gid_type const & split_gid)
    {
        HPX_ASSERT(is_preprocessing());
        buffer_->add_gid(gid, split_gid);
    }

    bool output_archive::has_gid(naming::gid_type const & gid)
    {
        HPX_ASSERT(is_preprocessing());
        return buffer_->has_gid(gid);
    }

    naming::gid_type output_archive::get_new_gid(naming::gid_type const & gid)
    {
        if(!splitted_gids_) return naming::gid_type();

        splitted_gids_type::iterator it = splitted_gids_->find(gid);
        HPX_ASSERT(it != splitted_gids_->end());
        HPX_ASSERT(it->second != naming::invalid_gid);
        naming::gid_type new_gid = it->second;
#if defined(HPX_DEBUG)
        it->second = naming::invalid_gid;
#endif
        return new_gid;
    }

}}
