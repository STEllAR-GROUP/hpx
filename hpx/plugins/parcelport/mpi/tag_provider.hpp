//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_TAG_PROVIDER_HPP
#define HPX_PARCELSET_POLICIES_MPI_TAG_PROVIDER_HPP

#include <hpx/lcos/local/spinlock.hpp>

#include <deque>

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    struct tag_provider
    {
        typedef lcos::local::spinlock mutex_type;

        tag_provider()
          : next_tag_(2)
        {}

        int acquire()
        {
            int tag = -1;
            boost::lock_guard<mutex_type> l(mtx_);
            if(free_tags_.empty())
            {
                HPX_ASSERT(next_tag_ < (std::numeric_limits<int>::max)());
                tag = next_tag_++;
            }
            else
            {
                tag = free_tags_.front();
                free_tags_.pop_front();
            }
            HPX_ASSERT(tag > 1);
            return tag;
        }

        void release(int tag)
        {
            HPX_ASSERT(tag > 1);
            boost::lock_guard<mutex_type> l(mtx_);
            HPX_ASSERT(tag <= next_tag_);

            if(tag == next_tag_) return;

            free_tags_.push_back(tag);
        }

        mutex_type mtx_;
        int next_tag_;
        std::deque<int> free_tags_;
    };
}}}}

#endif
