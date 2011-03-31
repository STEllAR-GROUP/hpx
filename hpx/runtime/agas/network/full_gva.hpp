////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_83DB815F_26D5_4525_AC5B_E702FBD886D4)
#define HPX_83DB815F_26D5_4525_AC5B_E702FBD886D4

#include <boost/io/ios_state.hpp>
#include <boost/cstdint.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>

#include <hpx/exception.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/safe_bool.hpp>

namespace hpx { namespace agas
{

template <typename Protocol>
struct full_gva 
{
    typedef typename traits::network::endpoint_type<Protocol>::type
        endpoint_type;

    typedef int component_type;
    typedef boost::uint64_t lva_type;

    typedef std::size_t count_type;
    typedef std::ptrdiff_t offset_type;

    full_gva()
      : endpoint(),
        type(components::component_invalid),
        base_lva(0),
        count(0),
        offset(0) {}

    full_gva(endpoint_type const& ep,
              component_type t = components::component_invalid, lva_type a = 0,
              count_type c = 0, offset_type o = 0)
      : endpoint(ep),
        type(t),
        base_lva(a),
        count(c),
        offset(o) {}

    full_gva(endpoint_type const& ep, component_type t, void* a,
              count_type c = 0, offset_type o = 0)
      : endpoint(ep),
        type(t),
        base_lva(reinterpret_cast<lva_type>(a)),
        count(c),
        offset(o) {}

    full_gva(lva_type a)
      : endpoint(),
        type(components::component_invalid),
        base_lva(a),
        count(0),
        offset(0) {}

    full_gva(void* a)
      : endpoint(),
        type(components::component_invalid), 
        base_lva(reinterpret_cast<lva_type>(a)),
        count(0),
        offset(0) {}

    bool operator==(full_gva const& rhs) const
    {
        return type     == rhs.type
            && base_lva == rhs.base_lva 
            && endpoint == rhs.endpoint
            && count    == rhs.count
            && offset   == rhs.offset;
    }

    bool operator!=(full_gva const& rhs) const
    { return !(*this == rhs); }

    endpoint_type endpoint;
    component_type type;
    lva_type base_lva;
    count_type count;
    offset_type offset; 

  private:
    friend class boost::serialization::access;

    template<class Archive>
    void save(Archive& ar, const unsigned int version) const
    { ar << endpoint << type << base_lva; }

    template<class Archive>
    void load(Archive& ar, const unsigned int version)
    {
        if (version > traits::serialization_version<Protocol>::value) {
            throw exception(version_too_new, 
                "trying to load full GVA with unknown version");
        }
        ar >> endpoint >> type >> base_lva; 
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

template <typename Char, typename Traits, typename Protocol>
inline std::basic_ostream<Char, Traits>&
operator<< (std::basic_ostream<Char, Traits>& os,
            full_gva<Protocol> const& addr)
{
    boost::io::ios_flags_saver ifs(os); 
    os << "(" << traits::network::name<Protocol>() << " "
       << addr.endpoint << " " 
       << components::get_component_type_name(addr.type) << " "
       << std::showbase << std::hex << addr.base_lva << " "
       << std::dec << addr.count << " "
       << std::hex << addr.offset << ")"; 
    return os;
} 

}}

#endif // HPX_83DB815F_26D5_4525_AC5B_E702FBD886D4

