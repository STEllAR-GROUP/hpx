//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2014-2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_INTRUSIVE_PTR_HPP
#define HPX_SERIALIZATION_INTRUSIVE_PTR_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/detail/pointer.hpp>

#include <boost/intrusive_ptr.hpp>

namespace hpx { namespace serialization
{
    template <typename T>
    void load(input_archive & ar, boost::intrusive_ptr<T> & ptr, unsigned)
    {
        detail::serialize_pointer_tracked(ar, ptr);
    }

    template <typename T>
    void save(output_archive & ar, const boost::intrusive_ptr<T>& ptr, unsigned)
    {
        detail::serialize_pointer_tracked(ar, ptr);
    }

    HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE((template <class T>),
            (boost::intrusive_ptr<T>));
}}

#endif
