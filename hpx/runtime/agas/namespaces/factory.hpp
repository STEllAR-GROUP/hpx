////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_D9951196_521D_4EA5_947D_43451437AEE6)
#define HPX_D9951196_521D_4EA5_947D_43451437AEE6

#include <vector>
#include <map>

#include <boost/iterator/transform_iterator.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/std_pair.hpp>

#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/basic_namespace.hpp>

namespace hpx { namespace agas // hpx::agas
{

namespace tag { struct factory_namespace; } // hpx::agas::tag

namespace traits { // hpx::agas::traits

template <>
struct namespace_name_hook<tag::factory_namespace>
{
    typedef char const* result_type;

    static result_type call()
    { return "factory"; }
};

template <>
struct registry_type<tag::factory_namespace>
{ typedef std::multimap<naming::gid_type, boost::uint32_t> type; };

template <>
struct bind_hook<tag::factory_namespace>
{
    typedef registry_type<tag::factory_namespace>::type registry_type;
    typedef key_type<tag::factory_namespace>::type key_type;
    typedef mapped_type<tag::factory_namespace>::type mapped_type;

    typedef key_type result_type;

    static result_type call(registry_type& reg, key_type const& key,
                            mapped_type const& value)
    { return reg.insert(registry_type::value_type(key, value))->first; }
};

template <>
struct resolve_hook<tag::factory_namespace>
{
    typedef registry_type<tag::factory_namespace>::type registry_type;
    typedef key_type<tag::factory_namespace>::type key_type;
    typedef mapped_type<tag::factory_namespace>::type mapped_type;

    typedef registry_type::iterator iterator;

    typedef std::vector<mapped_type> result_type;

    static result_type call(registry_type& reg, key_type const& key)
    {
        typedef registry_type::value_type at_type;
        boost::fusion::result_of::at_c<at_type, 1>::type (*at) (at_type&) =
          &boost::fusion::at_c<1, at_type>;

        std::pair<iterator, iterator> p = reg.equal_range(key);
        return result_type(
            boost::make_transform_iterator(p.first, at),
            boost::make_transform_iterator(p.second, at));
    }
};

} // hpx::agas::traits
} // hpx::agas

///////////////////////////////////////////////////////////////////////////////
namespace components { namespace agas // hpx::components::agas
{

struct factory_namespace
  : private basic_namespace<hpx::agas::tag::factory_namespace>
{
    // TODO: implement interface
};

// MPL metafunction (syntactic sugar)
template <typename Protocal>
struct factory_namespace_type
{ typedef factory_namespace type; }; 

///////////////////////////////////////////////////////////////////////////////
namespace server // hpx::components::agas::server
{

// MPL metafunction
template <typename Protocal>
struct factory_namespace_type
{ typedef server::basic_namespace<hpx::agas::tag::factory_namespace> type; };

} // hpx::components::agas::server

///////////////////////////////////////////////////////////////////////////////
namespace stubs // hpx::components::agas::stubs
{

// MPL metafunction
template <typename Protocal>
struct factory_namespace_type
{ typedef stubs::basic_namespace<hpx::agas::tag::factory_namespace> type; };

} // hpx::components::agas::stubs

} // hpx::components::agas
} // hpx::components
} // hpx

#endif // HPX_D9951196_521D_4EA5_947D_43451437AEE6

