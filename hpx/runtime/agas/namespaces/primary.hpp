////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_AD465B65_6482_4A9E_889D_3EC1FE2E9486)
#define HPX_AD465B65_6482_4A9E_889D_3EC1FE2E9486

#include <map>

#include <boost/fusion/include/at_c.hpp>

#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/local_address.hpp>
#include <hpx/runtime/agas/basic_namespace.hpp>

namespace hpx { namespace agas // hpx::agas
{

namespace tag { // hpx::agas::tag

template <typename Protocal>
struct primary_namespace;

} // hpx::agas::tag

namespace traits { // hpx::agas::traits

template <typename Protocal>
struct namespace_name_hook<tag::primary_namespace<Protocal> >
{
    typedef char const* result_type;

    static result_type call()
    {
        std::string tag_name = protocol_name<Protocal>();
        tag_name += "/primary";
        return tag_name.c_str();
    }
};

template <typename Protocal>
struct registry_type<tag::primary_namespace<Protocal> >
{
    typedef std::map<naming::gid_type,
        typename local_address<Protocal>::registry_entry_type
    > type;
};

// TODO: implement bind_hook, update_hook, resolve_hook and unbind_hook

template <typename Protocal>
struct bind_hook<tag::primary_namespace<Protocal> >
{
    typedef typename registry_type<tag::primary_namespace<Protocal> >::type
        registry_type;
    typedef typename key_type<tag::primary_namespace<Protocal> >::type
        key_type;
    typedef typename mapped_type<tag::primary_namespace<Protocal> >::type
        mapped_type;

    typedef key_type result_type;

    static result_type create(registry_type& reg, key_type const& id,
                              mapped_type const& value)
    {
        key_type upper_bound;
        upper_bound = id + (at_c<1>(value) - 1);

        if (id.get_msb() != upper_bound.get_msb())
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                BOOST_CURRENT_FUNCTION,
                "msb's of lower and upper GID range should match")
        }
        
        return (reg.insert
            (typename registry_type::value_type(id, value)).first)->first;
    }

    static result_type call(registry_type& reg, key_type const& key,
                            mapped_type const& value)
    {
        using boost::fusion::at_c;

        key_type id = key;
        naming::strip_credit_from_gid(id);

        typename registry_type::iterator it = reg.lower_bound(id);

        if (it != reg.end())
        {
            if ((*it).first == id)
            {
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    BOOST_CURRENT_FUNCTION,
                    "supplied key is already bound")
            }

            else if (it != reg.begin())
            {
                --it;
                // previous range covers the new id
                if ((*it).first + at_c<1>((*it).second) > id)
                {
                    HPX_THROW_EXCEPTION(hpx::bad_parameter,
                        BOOST_CURRENT_FUNCTION,
                        "supplied key is contained in an existing range")
                }
            }
        }

        else
        {
            if (!reg.empty()) {
                --it; 
                // previous range covers the new id
                if ((*it).first + at_c<1>((*it).second) > id)
                {
                    HPX_THROW_EXCEPTION(hpx::bad_parameter,
                        BOOST_CURRENT_FUNCTION,
                        "supplied key is contained in an existing range")
                }
            }
        }            
    
        return create(reg, id, value);
    }
};

} // hpx::agas::traits
} // hpx::agas

///////////////////////////////////////////////////////////////////////////////
namespace components { namespace agas // hpx::components::agas
{

// MPL metafunction
template <typename Protocal>
struct primary_namespace_type
{ typedef basic_namespace<hpx::agas::tag::primary_namespace<Protocal> > type; }

///////////////////////////////////////////////////////////////////////////////
namespace server // hpx::components::agas::server
{

// MPL metafunction
template <typename Protocal>
struct primary_namespace_type
{
    typedef server::basic_namespace<
        hpx::agas::tag::primary_namespace<Protocal>
    > type;
};

} // hpx::components::agas::server

///////////////////////////////////////////////////////////////////////////////
namespace stubs // hpx::components::agas::stubs
{

// MPL metafunction
template <typename Protocal>
struct primary_namespace_type
{
    typedef stubs::basic_namespace<
        hpx::agas::tag::primary_namespace<Protocal>
    > type;
};

} // hpx::components::agas::stubs

} // hpx::components::agas
} // hpx::components
} // hpx

#endif // HPX_AD465B65_6482_4A9E_889D_3EC1FE2E9486

