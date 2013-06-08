//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_VERIFY_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_VERIFY_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/name.hpp>

#include <vector>

#include "certificate_store.hpp"
#include "parcel_suffix.hpp"
#include "signed_type.hpp"

namespace hpx { namespace components { namespace security
{
    bool verify(certificate_store const & certificate_store,
                std::vector<char> const& parcel_data,
                naming::gid_type& parcel_id,
                error_code& ec = throws)
    {
        if (parcel_data.size() < sizeof(signed_type<parcel_suffix>))
        {
            HPX_THROWS_IF(
                ec
              , hpx::security_error
              , "verify"
              , "Parcel too short to contain a parcel_suffix"
            );
            return false;
        }

//        server::hash hash;
//        hash.update(
//            reinterpret_cast<unsigned char const*>(&parcel_data.front()),
//            parcel_data.size() - sizeof(signed_parcel_suffix));
//        hash.final();

        signed_type<parcel_suffix> const * parcel_suffix_ptr =
            reinterpret_cast<signed_type<parcel_suffix> const *>(
                &(parcel_data.back()) - signed_type<parcel_suffix>::size() + 1);

        parcel_id = parcel_suffix_ptr->get_type().get_parcel_id();

        hpx::naming::gid_type subject(
            hpx::naming::replace_locality_id(
                HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_MSB
              , get_locality_id_from_gid(parcel_id))
          , HPX_SUBORDINATE_CERTIFICATE_AUTHORITY_LSB);

        // std::cout << subject << "\n";

        // TODO

        if (&ec != &throws)
            ec = make_success_code();

        return true;
    }
}}}

#endif
