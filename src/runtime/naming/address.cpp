//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/address.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/serialization/array.hpp>

#include <boost/mpl/bool.hpp>

namespace hpx { namespace naming { namespace detail
{
    struct name_serialization_data
    {
        locality_serialization_data locality_;
        address::address_type address_;
        address::component_type type_;
    };
}}}

namespace boost { namespace serialization
{
    template <>
    struct is_bitwise_serializable<
            hpx::naming::detail::name_serialization_data>
       : boost::mpl::true_
    {};
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    template <typename Archive>
    void address::save(Archive& ar, const unsigned int version) const
    {
        if (ar.flags() & util::disable_array_optimization) {
            ar << locality_ << type_ << address_;
        }
        else {
            detail::name_serialization_data data;
            fill_serialization_data(locality_, data.locality_);
            data.type_ = type_;
            data.address_ = address_;
            ar << boost::serialization::make_array(&data, 1);
        }
    }

    template <typename Archive>
    void address::load(Archive& ar, const unsigned int version)
    {
        if (version > HPX_ADDRESS_VERSION)
        {
            HPX_THROW_EXCEPTION(version_too_new,
                "address::load",
                "trying to load address with unknown version");
        }

        if (ar.flags() & util::disable_array_optimization) {
            ar >> locality_ >> type_ >> address_;
        }
        else {
            detail::name_serialization_data data;
            ar >> boost::serialization::make_array(&data, 1);
            fill_from_serialization_data(data.locality_, locality_);
            type_ = data.type_;
            address_ = data.address_;
        }
    }

    template HPX_EXPORT
    void address::save(util::portable_binary_oarchive&, const unsigned int) const;

    template HPX_EXPORT
    void address::load(util::portable_binary_iarchive&, const unsigned int);
}}
