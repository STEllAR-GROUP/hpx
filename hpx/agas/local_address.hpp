////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_68F6AFA9_0C73_4CE8_8D9A_D34D53DA1DF1)
#define HPX_68F6AFA9_0C73_4CE8_8D9A_D34D53DA1DF1

#include <boost/io/ios_state.hpp>
#include <boost/cstdint.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>

#include <hpx/runtime/components/component_type.hpp>
#include <hpx/util/safe_bool.hpp>
#include <hpx/agas/magic.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
// local_address<> serialization format version
#define HPX_LOCAL_ADDRESS_VERSION 0x10

namespace hpx { namespace agas
{

template <typename Protocal>
struct local_address 
{
    typedef boost::fusion::vector<
        local_address<Protocal>, std::size_t, std::ptrdiff_t
    > registry_entry_type;

    // locality_type is usually an Asio endpoint
    typedef typename magic::locality_type<Protocal>::type locality_type;
    typedef boost::int64_t component_type;
    typedef boost::uint64_t lva_type;

    local_address()
      : _locality(), _type(components::component_invalid), _lva(0) {}

    local_address(locality_type const& l,
                  component_type t = components::component_invalid,
                  lva_type lva = 0)
      : _locality(l), _type(t), _lva(lva) {}

    local_address(locality_type const& l, component_type t, void* lva)
      : _locality(l), _type(t), _lva(reinterpret_cast<address_type>(lva)) {}

    local_address(lva_type a)
      : _locality(), _type(components::component_invalid), _lva(a) {}

    local_address(void* lva)
      : _locality(), _type(components::component_invalid), 
        _lva(reinterpret_cast<lva_type>(lva)) {}

    operator util::safe_bool<local_address>::result_type() const 
    { 
        return util::safe_bool<local_address>()
            (components::component_invalid != _type || 0 != _lva); 
    }

    bool operator==(local_address const& rhs)
    {
        return _type     == rhs._type
            && _lva      == rhs._lva 
            && _locality == rhs._locality;
    }

    locality_type locality() const
    { return _locality; }

    component_type type() const
    { return _type; }
    
    lva_type lva() const
    { return _lva; }

  private:
    locality_type _locality;
    component_type _type;
    lva_type _lva; 

    friend class boost::serialization::access;

    template<class Archive>
    void save(Archive& ar, const unsigned int version) const
    { ar << _locality << _type << _lva; }

    template<class Archive>
    void load(Archive& ar, const unsigned int version)
    {
        if (version > HPX_LOCAL_ADDRESS_VERSION) {
            throw exception(version_too_new, 
                "trying to load local_address with unknown version");
        }
        ar >> _locality >> _type >> _lva; 
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

template <typename Char, typename Traits, typename Protocal>
inline std::basic_ostream<Char, Traits>&
operator<< (std::basic_ostream<Char, Traits>& out,
            local_address<Protocal> const& addr)
{
    boost::io::ios_flags_saver ifs(os); 
    os << "(" << magic::protocal_name<Protocal>() << ":"
       << addr.locality() << ":" 
       << components::get_component_type_name((int)addr.type()) 
       << ":0x" << std::hex << addr.lva() << ")"; 
    return os;
} 

}}

//BOOST_CLASS_VERSION(hpx::agas::local_address, HPX_LOCAL_ADDRESS_VERSION)
//BOOST_CLASS_TRACKING(hpx::agas::local_address, boost::serialization::track_never)

#include <hpx/config/warnings_suffix.hpp>

#endif // HPX_68F6AFA9_0C73_4CE8_8D9A_D34D53DA1DF1

