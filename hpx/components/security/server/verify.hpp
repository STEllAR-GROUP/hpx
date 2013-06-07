//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_VERIFY_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_VERIFY_HPP

#include <boost/shared_ptr.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <vector>

#include "certificate_store.hpp"
#include "parcel_suffix.hpp"
#include "signed_type.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    bool verify(certificate_store const & certificate_store,
                boost::shared_ptr<std::vector<char> > parcel_data)
    {
        if (parcel_data->size() < sizeof(signed_type<parcel_suffix>))
        {
            HPX_THROW_EXCEPTION(
                hpx::security_error
              , "verify"
              , "Parcel too short to contain a parcel_suffix"
            )
        }

        signed_type<parcel_suffix> const * parcel_suffix_ptr =
            reinterpret_cast<signed_type<parcel_suffix> const *>(
                &(parcel_data->back()) - sizeof(signed_type<parcel_suffix>));

        hpx::naming::gid_type const & parcel_id =
            parcel_suffix_ptr->get_type().get_parcel_id();

        hpx::naming::gid_type subject(
            hpx::naming::replace_locality_id(
                HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_MSB
              , get_locality_id_from_gid(parcel_id))
          , HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_LSB);

        std::cout << subject << "\n";

        // TODO

        return true;
    }
}}}}

#endif
