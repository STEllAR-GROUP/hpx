////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_68F6AFA9_0C73_4CE8_8D9A_D34D53DA1DF1)
#define HPX_68F6AFA9_0C73_4CE8_8D9A_D34D53DA1DF1

#include <hpx/runtime/agas/network/full_gva.hpp>

namespace hpx { namespace agas
{

template <typename Protocol>
struct gva 
{
    typedef typename full_gva<Protocol>::endpoint_type endpoint_type;
    typedef typename full_gva<Protocol>::component_type component_type;
    typedef typename full_gva<Protocol>::lva_type lva_type;

    gva()
      : endpoint(),
        type(components::component_invalid),
        lva(0) {}

    gva(endpoint_type const& ep,
        component_type t = components::component_invalid, lva_type a = 0)
      : endpoint(ep),
        type(t),
        lva(a) {}

    gva(endpoint_type const& ep, component_type t, void* a)
      : endpoint(ep),
        type(t),
        lva(reinterpret_cast<lva_type>(a)) {}

    gva(lva_type a)
      : endpoint(),
        type(components::component_invalid),
        lva(a) {}

    gva(void* a)
      : endpoint(),
        type(components::component_invalid), 
        lva(reinterpret_cast<lva_type>(a)) {}

    gva(full_gva<Protocol> const& f, naming::gid_type const& gid,
         naming::gid_type const& gidbase)
      : endpoint(f.endpoint),
        type(f.type)
    { lva = (gid.get_lsb() - gidbase.get_lsb()) * f.offset; }

    operator typename util::safe_bool<gva>::result_type() const 
    { 
        return util::safe_bool<gva>()
            (components::component_invalid != type || 0 != lva); 
    }

    bool operator==(gva const& rhs) const
    {
        return type     == rhs.type
            && lva      == rhs.lva 
            && endpoint == rhs.endpoint;
    }

    bool operator!=(gva const& rhs) const
    { return !(*this == rhs); }

    endpoint_type endpoint;
    component_type type;
    lva_type lva; 

  private:
    friend class boost::serialization::access;

    template<class Archive>
    void save(Archive& ar, const unsigned int version) const
    { ar << endpoint << type << lva; }

    template<class Archive>
    void load(Archive& ar, const unsigned int version)
    {
        if (version > traits::serialization_version<Protocol>::value) {
            throw exception(version_too_new, 
                "trying to load GVA with unknown version");
        }
        ar >> endpoint >> type >> lva; 
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

template <typename Char, typename Traits, typename Protocol>
inline std::basic_ostream<Char, Traits>&
operator<< (std::basic_ostream<Char, Traits>& os, gva<Protocol> const& addr)
{
    boost::io::ios_flags_saver ifs(os); 
    os << "(" << traits::network::name<Protocol>() << " "
       << addr.endpoint << " " 
       << components::get_component_type_name(addr.type) << " "
       << std::showbase << std::hex << addr.lva << ")"; 
    return os;
} 

}}

#endif // HPX_68F6AFA9_0C73_4CE8_8D9A_D34D53DA1DF1

