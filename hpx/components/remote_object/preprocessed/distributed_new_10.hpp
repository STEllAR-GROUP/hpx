// Copyright (c) 2007-2012 Hartmut Kaiser
// Copyright (c)      2012 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace components {
    template <typename T>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        BOOST_FOREACH(naming::id_type const & prefix, prefixes)
        {
            std::size_t numcreate = objs_per_loc;
            if (excess != 0) {
                --excess;
                ++numcreate;
            }
            if (created_count + numcreate > count)
                numcreate = count - created_count;
            if (numcreate == 0)
                break;
            for (std::size_t i = 0; i < numcreate; ++i) {
                res.push_back(
                    new_<T>(prefix)
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
namespace hpx { namespace components {
    template <typename T, typename A0>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        BOOST_FOREACH(naming::id_type const & prefix, prefixes)
        {
            std::size_t numcreate = objs_per_loc;
            if (excess != 0) {
                --excess;
                ++numcreate;
            }
            if (created_count + numcreate > count)
                numcreate = count - created_count;
            if (numcreate == 0)
                break;
            for (std::size_t i = 0; i < numcreate; ++i) {
                res.push_back(
                    new_<T>(prefix, boost::forward<A0>( a0 ))
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
namespace hpx { namespace components {
    template <typename T, typename A0 , typename A1>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        BOOST_FOREACH(naming::id_type const & prefix, prefixes)
        {
            std::size_t numcreate = objs_per_loc;
            if (excess != 0) {
                --excess;
                ++numcreate;
            }
            if (created_count + numcreate > count)
                numcreate = count - created_count;
            if (numcreate == 0)
                break;
            for (std::size_t i = 0; i < numcreate; ++i) {
                res.push_back(
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ))
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
namespace hpx { namespace components {
    template <typename T, typename A0 , typename A1 , typename A2>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        BOOST_FOREACH(naming::id_type const & prefix, prefixes)
        {
            std::size_t numcreate = objs_per_loc;
            if (excess != 0) {
                --excess;
                ++numcreate;
            }
            if (created_count + numcreate > count)
                numcreate = count - created_count;
            if (numcreate == 0)
                break;
            for (std::size_t i = 0; i < numcreate; ++i) {
                res.push_back(
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ))
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
namespace hpx { namespace components {
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        BOOST_FOREACH(naming::id_type const & prefix, prefixes)
        {
            std::size_t numcreate = objs_per_loc;
            if (excess != 0) {
                --excess;
                ++numcreate;
            }
            if (created_count + numcreate > count)
                numcreate = count - created_count;
            if (numcreate == 0)
                break;
            for (std::size_t i = 0; i < numcreate; ++i) {
                res.push_back(
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ))
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
namespace hpx { namespace components {
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        BOOST_FOREACH(naming::id_type const & prefix, prefixes)
        {
            std::size_t numcreate = objs_per_loc;
            if (excess != 0) {
                --excess;
                ++numcreate;
            }
            if (created_count + numcreate > count)
                numcreate = count - created_count;
            if (numcreate == 0)
                break;
            for (std::size_t i = 0; i < numcreate; ++i) {
                res.push_back(
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ))
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
namespace hpx { namespace components {
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        BOOST_FOREACH(naming::id_type const & prefix, prefixes)
        {
            std::size_t numcreate = objs_per_loc;
            if (excess != 0) {
                --excess;
                ++numcreate;
            }
            if (created_count + numcreate > count)
                numcreate = count - created_count;
            if (numcreate == 0)
                break;
            for (std::size_t i = 0; i < numcreate; ++i) {
                res.push_back(
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ))
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
namespace hpx { namespace components {
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        BOOST_FOREACH(naming::id_type const & prefix, prefixes)
        {
            std::size_t numcreate = objs_per_loc;
            if (excess != 0) {
                --excess;
                ++numcreate;
            }
            if (created_count + numcreate > count)
                numcreate = count - created_count;
            if (numcreate == 0)
                break;
            for (std::size_t i = 0; i < numcreate; ++i) {
                res.push_back(
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ))
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
namespace hpx { namespace components {
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        BOOST_FOREACH(naming::id_type const & prefix, prefixes)
        {
            std::size_t numcreate = objs_per_loc;
            if (excess != 0) {
                --excess;
                ++numcreate;
            }
            if (created_count + numcreate > count)
                numcreate = count - created_count;
            if (numcreate == 0)
                break;
            for (std::size_t i = 0; i < numcreate; ++i) {
                res.push_back(
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ))
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
namespace hpx { namespace components {
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        BOOST_FOREACH(naming::id_type const & prefix, prefixes)
        {
            std::size_t numcreate = objs_per_loc;
            if (excess != 0) {
                --excess;
                ++numcreate;
            }
            if (created_count + numcreate > count)
                numcreate = count - created_count;
            if (numcreate == 0)
                break;
            for (std::size_t i = 0; i < numcreate; ++i) {
                res.push_back(
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ))
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
namespace hpx { namespace components {
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        BOOST_FOREACH(naming::id_type const & prefix, prefixes)
        {
            std::size_t numcreate = objs_per_loc;
            if (excess != 0) {
                --excess;
                ++numcreate;
            }
            if (created_count + numcreate > count)
                numcreate = count - created_count;
            if (numcreate == 0)
                break;
            for (std::size_t i = 0; i < numcreate; ++i) {
                res.push_back(
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ))
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
namespace hpx { namespace components {
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        BOOST_FOREACH(naming::id_type const & prefix, prefixes)
        {
            std::size_t numcreate = objs_per_loc;
            if (excess != 0) {
                --excess;
                ++numcreate;
            }
            if (created_count + numcreate > count)
                numcreate = count - created_count;
            if (numcreate == 0)
                break;
            for (std::size_t i = 0; i < numcreate; ++i) {
                res.push_back(
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ))
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
namespace hpx { namespace components {
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        BOOST_FOREACH(naming::id_type const & prefix, prefixes)
        {
            std::size_t numcreate = objs_per_loc;
            if (excess != 0) {
                --excess;
                ++numcreate;
            }
            if (created_count + numcreate > count)
                numcreate = count - created_count;
            if (numcreate == 0)
                break;
            for (std::size_t i = 0; i < numcreate; ++i) {
                res.push_back(
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ))
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
namespace hpx { namespace components {
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        BOOST_FOREACH(naming::id_type const & prefix, prefixes)
        {
            std::size_t numcreate = objs_per_loc;
            if (excess != 0) {
                --excess;
                ++numcreate;
            }
            if (created_count + numcreate > count)
                numcreate = count - created_count;
            if (numcreate == 0)
                break;
            for (std::size_t i = 0; i < numcreate; ++i) {
                res.push_back(
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ))
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
