//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_PARCEL_SUFFIX_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_PARCEL_SUFFIX_HPP

#include <boost/cstdint.hpp>
#include <boost/io/ios_state.hpp>
#include <hpx/hpx_fwd.hpp>

#include "hash.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    class parcel_suffix
    {
    public:
        parcel_suffix()
        {
        }

        parcel_suffix(naming::gid_type const& parcel_id, hash & hash)
          : parcel_id_(parcel_id), hash_(hash.final())
        {
        }

        naming::gid_type const & get_parcel_id() const
        {
            return parcel_id_;
        }

        traits::hash<>::final_type const & get_hash() const
        {
            return hash_;
        }

        friend std::ostream & operator<<(std::ostream & os,
                                         parcel_suffix const & parcel_suffix)
        {
            boost::io::ios_flags_saver ifs(os);

            os << "<parcel_suffix "
               << parcel_suffix.parcel_id_
               << " \"";

            for (std::size_t i = 0; i < traits::hash<>::final_type::static_size; ++i)
            {
                os << std::hex
                   << std::nouppercase
                   << std::setfill('0')
                   << std::setw(2)
                   << static_cast<unsigned int>(parcel_suffix.hash_[i]);
            }

            return os << "\">";
        }

    private:
        naming::gid_type parcel_id_;
        traits::hash<>::final_type hash_;
    };
}}}}

#endif
