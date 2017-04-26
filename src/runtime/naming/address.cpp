//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/naming/address.hpp>

#include <hpx/throw_exception.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

#include <boost/io/ios_state.hpp>

#include <iomanip>
#include <iostream>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    template <typename Archive>
    void address::save(Archive& ar, const unsigned int version) const
    {
        ar & locality_ & type_ & address_;
    }

    template <typename Archive>
    void address::load(Archive& ar, const unsigned int version)
    {
        ar & locality_ & type_ & address_;
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
