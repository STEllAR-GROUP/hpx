//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file base_object.hpp

#ifndef HPX_SERIALIZATION_BASE_OBJECT_HPP
#define HPX_SERIALIZATION_BASE_OBJECT_HPP

#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/runtime/serialization/access.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <hpx/traits/polymorphic_traits.hpp>
#include <hpx/util/decay.hpp>

namespace hpx { namespace serialization
{
    template <typename Derived, typename Base, typename Enable =
        typename hpx::traits::is_intrusive_polymorphic<Derived>::type>
    struct base_object_type
    {
        base_object_type(Derived & d) : d_(d) {}
        Derived & d_;

        template <class Archive>
        void serialize(Archive & ar, unsigned)
        {
            access::serialize(ar, static_cast<Base&>(
                const_cast<typename hpx::util::decay<Derived>::type&>(d_)), 0);
        }
    };

    // we need another specialization to explicitly
    // specify non-virtual calls of virtual functions in
    // intrusively serialized base classes.
    template <typename Derived, typename Base>
    struct base_object_type<Derived, Base, boost::mpl::true_>
    {
        base_object_type(Derived & d) : d_(d) {}
        Derived & d_;

        template <class Archive>
        void save(Archive & ar, unsigned) const
        {
            access::save_base_object(ar, static_cast<const Base&>(d_), 0);
        }

        template <class Archive>
        void load(Archive & ar, unsigned)
        {
            access::load_base_object(ar, static_cast<Base&>(d_), 0);
        }
        HPX_SERIALIZATION_SPLIT_MEMBER();
    };

    template <typename Base, typename Derived>
    base_object_type<Derived, Base> base_object(Derived & d)
    {
        return base_object_type<Derived, Base>(d);
    }

    // allow our base_object_type to be serialized as prvalue
    // compiler should support good ADL implementation
    // but it is rather for all hpx serialization library
    template <typename D, typename B> BOOST_FORCEINLINE
    output_archive & operator<<(output_archive & ar, base_object_type<D, B> t)
    {
        ar.invoke(t);
        return ar;
    }

    template <typename D, typename B> BOOST_FORCEINLINE
    input_archive & operator>>(input_archive & ar, base_object_type<D, B> t)
    {
        ar.invoke(t);
        return ar;
    }

    template <typename D, typename B> BOOST_FORCEINLINE
    output_archive & operator&(output_archive & ar, base_object_type<D, B> t) //-V524
    {
        ar.invoke(t);
        return ar;
    }

    template <typename D, typename B> BOOST_FORCEINLINE
    input_archive & operator&(input_archive & ar, base_object_type<D, B> t) //-V524
    {
        ar.invoke(t);
        return ar;
    }
}}

#endif
