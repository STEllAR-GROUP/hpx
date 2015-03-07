//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2014-2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_SERIALIZE_HPP
#define HPX_SERIALIZATION_SERIALIZE_HPP

#include <hpx/config.hpp>

#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/access.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>

#include <boost/type_traits/add_const.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace serialization {

    template <typename Archive, typename T>
    void serialize(Archive & ar, T & t, unsigned)
    {
        access::serialize(ar, t, 0);
    }

    template <typename T>
    output_archive & operator<<(output_archive & ar, T const & t)
    {
        ar.invoke(t);
        return ar;
    }

    template <typename T>
    input_archive & operator>>(input_archive & ar, T & t)
    {
        ar.invoke(t);
        return ar;
    }

    template <typename T>
    output_archive & operator&(output_archive & ar, T const & t)
    {
        ar.invoke(t);
        return ar;
    }

    template <typename T>
    input_archive & operator&(input_archive & ar, T & t)
    {
        ar.invoke(t);
        return ar;
    }
}}

#include <hpx/serialization/base_object.hpp>
#include <hpx/serialization/polymorphic_nonintrusive_factory.ipp>

#endif
