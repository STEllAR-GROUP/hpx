//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/naming/address.hpp>

#include <hpx/throw_exception.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <boost/io/ios_state.hpp>

#include <iomanip>
#include <iostream>

namespace hpx { namespace naming { namespace detail
{
    struct name_serialization_data
    {
        gid_type locality_;
        address::component_type type_;
        address::address_type address_;
        address::address_type offset_;

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            ar & locality_ & type_ & address_ & offset_;
        }
    };
}}}

HPX_IS_BITWISE_SERIALIZABLE(hpx::naming::detail::name_serialization_data)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    template <typename Archive>
    void address::save(Archive& ar, const unsigned int version) const
    {
        detail::name_serialization_data data{
            locality_, type_, address_, offset_};
        ar << data;
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

        detail::name_serialization_data data;
        ar >> data;
        locality_ = data.locality_;
        type_ = data.type_;
        address_ = data.address_;
        offset_ = data.offset_;
    }

    template HPX_EXPORT
    void address::save(serialization::output_archive&, const unsigned int) const;

    template HPX_EXPORT
    void address::load(serialization::input_archive&, const unsigned int);

    std::ostream& operator<<(std::ostream& os, address const& addr)
    {
        boost::io::ios_flags_saver ifs(os);
        os << "(" << addr.locality_ << ":"
           << components::get_component_type_name(addr.type_)
           << ":" << std::showbase << std::hex << addr.address_ << ")";
        return os;
    }
}}
