// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace components {
    template <typename T>
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ))
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
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ))
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
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ))
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
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ))
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
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ))
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
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ))
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
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ))
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
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ))
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
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ))
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
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ))
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
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ))
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
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ))
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
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15>
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16>
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17>
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18>
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17 , A18 && a18)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ) , std::forward<A18>( a18 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19>
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17 , A18 && a18 , A19 && a19)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ) , std::forward<A18>( a18 ) , std::forward<A19>( a19 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20>
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17 , A18 && a18 , A19 && a19 , A20 && a20)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ) , std::forward<A18>( a18 ) , std::forward<A19>( a19 ) , std::forward<A20>( a20 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21>
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17 , A18 && a18 , A19 && a19 , A20 && a20 , A21 && a21)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ) , std::forward<A18>( a18 ) , std::forward<A19>( a19 ) , std::forward<A20>( a20 ) , std::forward<A21>( a21 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21 , typename A22>
    std::vector<lcos::unique_future<object<T> > >
    distributed_new(std::size_t count, A0 && a0 , A1 && a1 , A2 && a2 , A3 && a3 , A4 && a4 , A5 && a5 , A6 && a6 , A7 && a7 , A8 && a8 , A9 && a9 , A10 && a10 , A11 && a11 , A12 && a12 , A13 && a13 , A14 && a14 , A15 && a15 , A16 && a16 , A17 && a17 , A18 && a18 , A19 && a19 , A20 && a20 , A21 && a21 , A22 && a22)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::unique_future<object<T> > > res;
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
                    new_<T>(prefix, std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ) , std::forward<A18>( a18 ) , std::forward<A19>( a19 ) , std::forward<A20>( a20 ) , std::forward<A21>( a21 ) , std::forward<A22>( a22 ))
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
