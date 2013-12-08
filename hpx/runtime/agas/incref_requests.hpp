//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_AGAS_INCREF_REQUESTS_DEC_01_203_0133PM)
#define HPX_RUNTIME_AGAS_INCREF_REQUESTS_DEC_01_203_0133PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <map>

namespace hpx { namespace agas { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    struct incref_request_data
    {
        incref_request_data(boost::uint64_t credit, boost::int32_t loc)
            : credit_(credit), keep_alive_(), locality_(loc), debit_(0)
        {}
        incref_request_data(boost::uint64_t credit, naming::id_type const& id,
                boost::int32_t loc = naming::invalid_locality_id)
            : credit_(credit), keep_alive_(id), locality_(loc), debit_(0)
        {}

        boost::int64_t credit_;         // amount of credit not acknowledged yet
        naming::id_type keep_alive_;    // id for which credit is outstanding
        boost::int32_t locality_;       // locality where this credit belongs
        boost::int64_t debit_;          // amount of credits held back because of
                                        // pending acknowledgments
    };

    ///////////////////////////////////////////////////////////////////////////
    class incref_requests
    {
    private:
        typedef lcos::local::spinlock mutex_type;

        typedef std::multimap<naming::gid_type, incref_request_data>
            incref_requests_type;

        typedef incref_requests_type::iterator iterator;
        typedef incref_requests_type::const_iterator const_iterator;

        iterator find_entry(naming::gid_type const& gid,
            boost::int32_t loc = naming::invalid_locality_id);

    public:
        incref_requests() {}

        // Add a credit request for a local id. This will add the credit to any
        // existing local request.
        //
        // This function will be called as part of setting up an incref request.
        // It will also keep the given id alive until all of the credits are
        // acknowledged (see acknowledge_request below).
        void add_incref_request(boost::int64_t credit, naming::id_type const& id);

        // Add a credit request for a remote id, this will subtract the credit from an
        // existing local request before adding the remote one. This function does
        // nothing if there is no local request for the given gid.
        //
        // This function will be called during id-splitting to store a bread-crumb
        // pointing to the locality where the outstanding credit has to be tracked.
        bool add_remote_incref_request(boost::int64_t credit,
            naming::gid_type const& gid, boost::int32_t remote_locality);

        // This function will be called whenever a acknowledgment message from AGAS
        // is received. It will compensate the pending credits for the acknowledged
        // amount of credits.
        typedef HPX_STD_FUNCTION<
            hpx::future<bool>(boost::int64_t, naming::gid_type const&, boost::int32_t)
        > acknowledge_request_callback;

        bool acknowledge_request(boost::int64_t credit, naming::id_type const& id,
             acknowledge_request_callback const& f);

        //
        bool add_decref_request(boost::int64_t credit, naming::gid_type const& id);

    private:
        mutable mutex_type mtx_;
        incref_requests_type store_;
    };
}}}

#endif
