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
            locality_.save(ar, HPX_LOCALITY_VERSION);

            detail::name_serialization_data data;
            data.type_ = type_;
            data.address_ = address_;
            ar.save(data);
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
            locality_.load(ar, HPX_LOCALITY_VERSION);

            detail::name_serialization_data data;
            ar.load(data);
            type_ = data.type_;
            address_ = data.address_;
        }
    }

    template HPX_EXPORT
    void address::save(util::portable_binary_oarchive&, const unsigned int) const;

    template HPX_EXPORT
    void address::load(util::portable_binary_iarchive&, const unsigned int);
}}
