////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Vinay C Amatya
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_5Z23KM0QLFNHPFQT9S9Q)
#define HPX_5Z23KM0QLFNHPFQT9S9Q


#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
//#include <hpx/runtime/agas/server/route.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>

namespace hpx { namespace agas
{

namespace server
{

bool primary_namespace::route(
    parcelset::parcel const& p
  , error_code& ec
  )
{
    // protection of parcel's content?
    // resolve id
    //error_code ec2;
    bool resolved;
    naming::address addr;
    naming::gid_type id = p.get_destination();
    //if(id)
    //    std::cout << "Parcel received" << std::endl;
    naming::strip_credit_from_gid(id);
    mutex_type::scoped_lock l(mutex_);
    // parcel action should set priority of thread.
    gva_table_type::const_iterator it = gvas_.lower_bound(id)
                                  , begin = gvas_.begin()
                                  , end = gvas_.end();

    if (it != end)
    {
        if(it->first  == id)
        {
            if (&ec != &throws)
                ec = make_success_code();
            const gva g = it->second;
            {
                addr.locality_ = g.endpoint;
                addr.type_ = g.type;
                addr.address_ = g.lva();
            }
        resolved = true;
            
        }
        else if (it != begin)
        {
            --it;
            if((it->first + it->second.count) > id)
            {
                if(HPX_UNLIKELY(id.get_msb() != it->first.get_msb()))
                {
                    HPX_THROWS_IF(ec, internal_server_error
                      , "primary_namespace::resolve_gid" 
                      , "MSBs of lower and upper range bound do not match");
                    //return response();
                }
                if (&ec != &throws)
                    ec = make_success_code();

                const gva g = it->second.resolve(id, it->first);
                {
                    addr.locality_ = g.endpoint;
                    addr.type_ = g.type;
                    addr.address_ = g.lva();
                }
            resolved = true;
            }

        }

    }
    else if (HPX_LIKELY(!gvas_.empty()))
    {
        --it;

        if((it->first + it->second.count) > id)
        {
            if (HPX_UNLIKELY(id.get_msb() != it->first.get_msb()))
            {
                HPX_THROWS_IF(ec, internal_server_error
                  , "primary_namespace::resolve_gid" 
                  , "MSBs of lower and upper range bound do not match");
                //return response();
            }

            if (&ec != &throws)
                ec = make_success_code();

                const gva g = it->second.resolve(id, it->first);
                {
                    addr.locality_ = g.endpoint;
                    addr.type_ = g.type;
                    addr.address_ = g.lva();
                }
                resolved = true;
        }
    }

    parcelset::parcel p_temp = p;
    p_temp.set_destination(id);
    p_temp.set_destination_addr(addr);
    //assign the parcel to parcelhandler
    hpx::applier::get_applier().get_parcel_handler().put_parcel(p_temp);
    //TO DO: update sender's cache for the destination id rosolution

    if(resolved)
        return true;
    return false;
}

}}}

#endif // HPX_5Z23KM0QLFNHPFQT9S9Q