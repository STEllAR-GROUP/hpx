//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2014-2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_SERIALIZE_HPP
#define HPX_SERIALIZATION_SERIALIZE_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/access.hpp>
#include <hpx/runtime/serialization/detail/size_gatherer_container.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <boost/type_traits/is_same.hpp>

namespace hpx { namespace serialization
{
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
    output_archive & operator&(output_archive & ar, T const & t) //-V524
    {
        ar.invoke(t);
        return ar;
    }

    template <typename T>
    input_archive & operator&(input_archive & ar, T & t) //-V524
    {
        ar.invoke(t);
        return ar;
    }
}}

#include <hpx/runtime/serialization/detail/polymorphic_nonintrusive_factory_impl.hpp>

#endif
