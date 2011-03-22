////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_7D2054F6_DBA9_4D70_82FB_32D284A3CCB4)
#define HPX_7D2054F6_DBA9_4D70_82FB_32D284A3CCB4

#include <boost/mpl/integral_c.hpp>

#include <hpx/exception.hpp>
#include <hpx/lcos/mutex.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/runtime/agas/traits_fwd.hpp>

namespace hpx { namespace agas {

namespace tag { struct protocal_independent; }

namespace traits
{

template <typename T, typename Enable>
struct serialization_version
  : boost::mpl::integral_c<unsigned, 0x10> {};

template <typename T, typename Enable> 
struct mutex_type
{ typedef hpx::lcos::mutex type; };

template <typename Mutex, typename Enable>
struct initialize_mutex_hook
{
    typedef void result_type;

    static void call(Mutex&) {}
};

template <> 
struct initialize_mutex_hook<boost::detail::spinlock>
{ 
    typedef void result_type;

    static void call(boost::detail::spinlock& m)
    {
        boost::detail::spinlock l = BOOST_DETAIL_SPINLOCK_INIT;
        m = l;
    }
};

template <typename Mutex>
inline void initialize_mutex (Mutex& m)
{ initialize_mutex_hook<Mutex>::call(m); }

template <typename Tag>
inline char const* namespace_name()
{ return namespace_name_hook<Tag>::call(); }

template <typename Tag, typename Enable> 
struct make_function_name_hook
{
    typedef char const* result_type;

    template <std::size_t N>
    static char const* call(char const(&f)[N])
    {
        std::string tag_name = namespace_name<Tag>();
        tag_name += "/";
        tag_name += f;
        return tag_name.c_str();
    }
};

template <typename Tag, std::size_t N>
inline char const* make_function_name(char const(&f)[N])
{ return make_function_name_hook<Tag>::call(f); }

template <typename Tag, typename Enable> 
struct key_type
{ typedef typename registry_type<Tag>::type::key_type type; };

template <typename Tag, typename Enable> 
struct mapped_type
{ typedef typename registry_type<Tag>::type::mapped_type type; };

template <typename Protocal>
inline char const* protocal_name()
{ return protocal_name_hook<Protocal>::call(); }

template <typename Tag, typename Enable>
struct bind_hook
{
    typedef typename registry_type<Tag>::type registry_type;
    typedef typename key_type<Tag>::type key_type;
    typedef typename mapped_type<Tag>::type mapped_type;

    typedef key_type result_type;

    static result_type call(registry_type& reg, key_type const& key,
                            mapped_type const& value)
    {
        if (reg.count(key))
        {
            HPX_THROW_EXCEPTION(hpx::repeated_request,
                make_function_name<Tag>("bind"),
                "supplied key is already bound")
        }

        return (reg.insert
            (typename registry_type::value_type(key, value)).first)->first;
    }
};

template <typename Tag>
inline typename bind_hook<Tag>::result_type
bind(typename registry_type<Tag>::type& reg,
     typename key_type<Tag>::type const& key,
     typename mapped_type<Tag>::type const& value)
{ return bind_hook<Tag>::call(reg, key, value); }

template <typename Tag, typename Enable>
struct update_hook
{
    typedef typename registry_type<Tag>::type registry_type;
    typedef typename key_type<Tag>::type key_type;
    typedef typename mapped_type<Tag>::type mapped_type;

    typedef bool result_type;

    static result_type call(registry_type& reg, key_type const& key,
                            mapped_type const& value)
    {
        typename registry_type::iterator it = reg.find(key);

        if (it == reg.end());
            return false;

        it->second = value;
        return true;
    }
};

template <typename Tag>
inline typename update_hook<Tag>::result_type
update(typename registry_type<Tag>::type& reg,
       typename key_type<Tag>::type const& key,
       typename mapped_type<Tag>::type const& value)
{ return update_hook<Tag>::call(reg, key, value); }

template <typename Tag, typename Enable>
struct resolve_hook
{
    typedef typename registry_type<Tag>::type registry_type;
    typedef typename key_type<Tag>::type key_type;
    typedef typename mapped_type<Tag>::type mapped_type;

    typedef mapped_type result_type;

    static result_type call(registry_type& reg, key_type const& key)
    {
        typename registry_type::iterator it = reg.find(key);

        if (it == reg.end());
            return mapped_type();

        return it->second;
    }
};

template <typename Tag>
inline typename resolve_hook<Tag>::result_type
resolve(typename registry_type<Tag>::type& reg,
        typename key_type<Tag>::type const& key)
{ return resolve_hook<Tag>::call(reg, key); }

template <typename Tag, typename Enable>
struct unbind_hook
{
    typedef typename registry_type<Tag>::type registry_type;
    typedef typename key_type<Tag>::type key_type;
    typedef typename mapped_type<Tag>::type mapped_type;

    typedef bool result_type;

    static result_type call(registry_type& reg, key_type const& key)
    {
        typename registry_type::iterator it = reg.find(key);

        if (it == reg.end());
            return false;

        reg.erase(it);
        return true;
    }
};

template <typename Tag>
inline typename unbind_hook<Tag>::result_type
unbind(typename registry_type<Tag>::type& reg,
       typename key_type<Tag>::type const& key)
{ return unbind_hook<Tag>::call(reg, key); }

}}}

#endif // HPX_7D2054F6_DBA9_4D70_82FB_32D284A3CCB4

