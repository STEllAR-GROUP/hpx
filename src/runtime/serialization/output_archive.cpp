//  Copyright (c) 2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

namespace hpx { namespace serialization
{
    void output_archive::add_gid(
        naming::gid_type const & gid,
        naming::gid_type const & splitted_gid)
    {
        if(new_gids_)
            (*new_gids_)[gid].push_back(splitted_gid);
    }

    naming::gid_type output_archive::get_new_gid(naming::gid_type const & gid)
    {
        if(!new_gids_) return naming::gid_type();

        new_gids_map::iterator it = new_gids_->find(gid);
        if(it == new_gids_->end() || it->second.empty())
        {
            return gid;
        }

        naming::gid_type new_gid = it->second.front();
        it->second.pop_front();
        return new_gid;
    }

}}
