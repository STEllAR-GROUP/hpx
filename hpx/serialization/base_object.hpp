//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file base_object.hpp

#ifndef HPX_SERIALIZATION_BASE_OBJECT_HPP
#define HPX_SERIALIZATION_BASE_OBJECT_HPP

#include <hpx/config.hpp>

#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/string.hpp>

namespace hpx { namespace serialization {
    template <typename T>
    const char * get_name();

    template <typename Derived, typename Base>
    struct base_object_type
    {
        base_object_type(Derived & d) : d_(d) {}
        Derived & d_;

        void save(output_archive & ar, unsigned) const
        {
            std::string name(get_name<Derived>());
            std::cout << "saving name " << name << "\n";
            ar << name;
            //d_.Base::serialize(ar, 0);
        }

        void load(input_archive & ar, unsigned)
        {
            //d_.Base::serialize(ar, 0);
        }

        HPX_SERIALIZATION_SPLIT_MEMBER();
    };

    template <typename Base, typename Derived>
    base_object_type<Derived, Base> base_object(Derived & d)
    {
        return base_object_type<Derived, Base>(d);
    }
}}

#endif
