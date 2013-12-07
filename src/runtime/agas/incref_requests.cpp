//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/incref_requests.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/lcos/future.hpp>

#include <boost/foreach.hpp>

namespace hpx { namespace agas { namespace detail
{
    incref_requests::iterator incref_requests::find_entry(
        naming::gid_type const& gid, naming::id_type const& loc)
    {
        std::pair<iterator, iterator> r = store_.equal_range(gid);

        for (/**/; r.first != r.second; ++r.first)
        {
            if (r.first->second.locality_ == loc)
            {
                return r.first;
            }
        }

        return store_.end();
    }

    void incref_requests::add_incref_request(boost::int64_t credit,
        naming::id_type const& id)
    {
        HPX_ASSERT(credit > 0);

        naming::gid_type gid = naming::detail::get_stripped_gid(id.get_gid());

        mutex_type::scoped_lock l(mtx_);

        // Either add the given credit to the local entry or create
        // a new entry.
        iterator it = find_entry(gid, naming::invalid_id);
        if (it != store_.end())
        {
            it->second.credit_ += credit;
            if (it->second.credit_ == 0)
            {
                // this credit request was already acknowledged,
                // remove the entry
                store_.erase(it);
            }
            else if (it->second.keep_alive_.get_management_type() ==
                     naming::id_type::unmanaged)
            {
                // If the currently stored id is unmanaged then the entry
                // was created by a remote acknowledgment. Update the
                // id with a local one to be kept alive.
                it->second.keep_alive_ = id;
            }
        }
        else
        {
            store_.insert(incref_requests_type::value_type(
                gid, incref_request_data(credit, id, naming::invalid_id)));
        }
    }

    bool incref_requests::add_remote_incref_request(boost::int64_t credit,
        naming::gid_type const& gid, naming::id_type const& remote_locality)
    {
        HPX_ASSERT(credit > 0);

        mutex_type::scoped_lock l(mtx_);

        // There is nothing for us to do if no local entry exists.
        iterator it_local = find_entry(gid, naming::invalid_id);
        if (it_local == store_.end())
            return false;

        // Subtract the given credit from any existing local entry, remove
        // the entry if the remaining outstanding credit becomes zero.
        HPX_ASSERT(it_local->second.credit_ >= credit);
        it_local->second.credit_ -= credit;
        if (it_local->second.credit_ == 0)
        {
            store_.erase(it_local);
            return false;
        }

        // Add the given credit to any existing entry or create new one.
        iterator it_remote = find_entry(gid, remote_locality);
        if (it_remote != store_.end())
        {
            it_remote->second.credit_ += credit;
            HPX_ASSERT(it_remote->second.credit_ != 0);
        }
        else
        {
            store_.insert(incref_requests_type::value_type(
                gid, incref_request_data(
                    credit, naming::invalid_id, remote_locality)));
        }

        return true;
    }

    bool incref_requests::acknowledge_request(boost::int64_t credit,
        naming::id_type const& id, acknowledge_request_callback const& f)
    {
        HPX_ASSERT(credit > 0);
        naming::gid_type gid = naming::detail::get_stripped_gid(id.get_gid());

        std::vector<incref_request_data> matching_data;

        {
            mutex_type::scoped_lock l(mtx_);

            std::pair<iterator, iterator> r = store_.equal_range(gid);
            while (r.first != r.second && credit != 0)
            {
                incref_request_data& data = r.first->second;
                if (data.credit_ > credit)
                {
                    // adjust remaining part of the credit
                    data.credit_ -= credit;

                    // construct proper acknowledgment data
                    matching_data.push_back(data);
                    matching_data.back().credit_ = credit;
                    matching_data.back().debit_ = 0;

                    // we're done with handling acknowledged credit
                    credit = 0;

                    ++r.first;
                }
                else
                {
                    // construct proper acknowledgment data
                    matching_data.push_back(data);

                    // adjust credit
                    credit -= data.credit_;

                    // delete the entry as it has been handled completely
                    iterator it = r.first++;
                    store_.erase(it);
                }
            }
            HPX_ASSERT(credit >= 0);

            if (credit != 0)
            {
                // add negative credit entry, we expect for a request to come
                // in shortly
                store_.insert(incref_requests_type::value_type(
                    gid, incref_request_data(-credit, id, naming::invalid_id)));
            }
        }

        // now handle all acknowledged credits
        std::vector<hpx::future<bool> > requests;
        requests.reserve(matching_data.size());

        BOOST_FOREACH(incref_request_data const& data, matching_data)
        {
            if (data.debit_ != 0)
            {
                HPX_ASSERT(data.locality_ == naming::invalid_id);
                HPX_ASSERT(data.debit_ > 0);
                requests.push_back(
                    f(-data.debit_, data.keep_alive_, data.locality_)
                );
            }

            if (data.locality_ != naming::invalid_id)
            {
                requests.push_back(
                    f(data.credit_, data.keep_alive_, data.locality_)
                );
            }
        }

        bool result = true;
        BOOST_FOREACH(hpx::future<bool>& f, requests)
        {
            result = f.get() && result;
        }

        return result;
    }

    bool incref_requests::add_decref_request(boost::int64_t credit,
        naming::gid_type const& gid)
    {
        mutex_type::scoped_lock l(mtx_);

        // There is nothing for us to do if no local entry exists.
        iterator it_local = find_entry(gid, naming::invalid_id);
        if (it_local == store_.end())
            return false;   // perform 'normal' (non-delayed) decref handling

        (*it_local).second.debit_ += credit;
        return true;
    }
}}}

