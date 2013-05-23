//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_STORE_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_CERTIFICATE_STORE_HPP

#include <boost/optional.hpp>
#include <hpx/hpx_fwd.hpp>

#include "certificate.hpp"
#include "identity.hpp"
#include "signed_type.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    class certificate_store
    {
        typedef std::map<identity, signed_type<certificate> > store_type;

    public:
        bool insert(signed_type<certificate> const & signed_certificate)
        {
            return store_.insert(
                std::make_pair(
                    signed_certificate.get_type().get_subject()
                  , signed_certificate)
            ).second;
        }

        boost::optional<signed_type<certificate> >
        at(identity const & subject) const
        {
            store_type::const_iterator iterator = store_.find(subject);

            if (iterator != store_.end())
                return iterator->second;

            return boost::none;
        }

    private:
        store_type store_;
    };
}}}}

#endif
