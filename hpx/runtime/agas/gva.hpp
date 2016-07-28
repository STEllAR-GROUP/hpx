////////////////////////////////////////////////////////////////////////////////
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_83DB815F_26D5_4525_AC5B_E702FBD886D4)
#define HPX_83DB815F_26D5_4525_AC5B_E702FBD886D4

#include <hpx/config.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/throw_exception.hpp>

#include <boost/cstdint.hpp>
#include <boost/io/ios_state.hpp>

namespace hpx { namespace agas
{

struct gva
{
    typedef boost::int32_t component_type;
    typedef boost::uint64_t lva_type;

    gva()
      : type(components::component_invalid),
        count(0),
        lva_(0),
        offset(0) {}

    explicit gva(naming::gid_type const& p,
        component_type t = components::component_invalid, boost::uint64_t c = 1,
        lva_type a = 0, boost::uint64_t o = 0)
      : prefix(p),
        type(t),
        count(c),
        lva_(a),
        offset(o) {}

    gva(naming::gid_type const& p, component_type t, boost::uint64_t c, void* a,
            boost::uint64_t o = 0)
      : prefix(p),
        type(t),
        count(c),
        lva_(reinterpret_cast<lva_type>(a)),
        offset(o) {}

    explicit gva(naming::gid_type const& p, lva_type a, component_type t)
      : prefix(p),
        type(t),
        count(1),
        lva_(a),
        offset(0) {}

    explicit gva(lva_type a, component_type t = components::component_invalid)
      : type(t),
        count(1),
        lva_(a),
        offset(0) {}

    explicit gva(void* a)
      : type(components::component_invalid),
        count(0),
        lva_(reinterpret_cast<lva_type>(a)),
        offset(0) {}

    gva& operator=(lva_type a)
    {
        prefix = naming::gid_type();
        type = components::component_invalid;
        count = 0;
        lva_ = a;
        offset = 0;
        return *this;
    }

    gva& operator=(void* a)
    {
        prefix = naming::gid_type();
        type = components::component_invalid;
        count = 0;
        lva_ = reinterpret_cast<lva_type>(a);
        offset = 0;
        return *this;
    }

    bool operator==(gva const& rhs) const
    {
        return type   == rhs.type
            && count  == rhs.count
            && lva_   == rhs.lva_
            && offset == rhs.offset
            && prefix == rhs.prefix;
    }

    bool operator!=(gva const& rhs) const
    { return !(*this == rhs); }

    void lva(lva_type a)
    { lva_ = a; }

    void lva(void* a)
    { lva_ = reinterpret_cast<lva_type>(a); }

    lva_type lva() const
    {
        return lva_;
    }

    lva_type lva(naming::gid_type const& gid,
                 naming::gid_type const& gidbase) const
    {
        lva_type l = lva_;
        l += (gid.get_lsb() - gidbase.get_lsb()) * offset;
        return l;
    }

    gva resolve(naming::gid_type const& gid,
                naming::gid_type const& gidbase) const
    {
        gva g(*this);
        g.lva_ = g.lva(gid, gidbase);

        // This is a hack to make sure that if resolve() or lva() is called on
        // the returned GVA, an exact copy will be returned (see the last two
        // lines of lva() above.
        g.count = 1;
        return g;
    }

    naming::gid_type prefix;
    component_type type;
    boost::uint64_t count;

  private:
    lva_type lva_;

  public:
    boost::uint64_t offset;

  private:
    friend class hpx::serialization::access;

    template<class Archive>
    void save(Archive& ar, const unsigned int version) const
    { ar << prefix << type << count << lva_ << offset; } //-V128

    template<class Archive>
    void load(Archive& ar, const unsigned int version)
    {
        if (version > HPX_AGAS_VERSION)
            HPX_THROW_EXCEPTION(version_too_new
              , "gva::load"
              , "trying to load GVA with unknown version");
        ar >> prefix >> type >> count >> lva_ >> offset; //-V128
    }

    HPX_SERIALIZATION_SPLIT_MEMBER()
};

template <typename Char, typename Traits>
inline std::basic_ostream<Char, Traits>&
operator<< (std::basic_ostream<Char, Traits>& os, gva const& addr)
{
    boost::io::ios_flags_saver ifs(os);
    os << "(" << addr.prefix << " "
       << components::get_component_type_name(addr.type) << " "
       << addr.count << " "
       << std::showbase << std::hex << addr.lva() << " "
       << addr.offset << ")";
    return os;
}

}}

#endif // HPX_83DB815F_26D5_4525_AC5B_E702FBD886D4

