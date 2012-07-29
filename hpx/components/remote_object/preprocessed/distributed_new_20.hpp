// Copyright (c) 2007-2012 Hartmut Kaiser
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
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
namespace hpx { namespace components {
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ))
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
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ))
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
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ))
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
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ))
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
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ))
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
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ))
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
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 ))
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
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19 , BOOST_FWD_REF(A20) a20)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 ) , boost::forward<A20>( a20 ))
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
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19 , BOOST_FWD_REF(A20) a20 , BOOST_FWD_REF(A21) a21)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 ) , boost::forward<A20>( a20 ) , boost::forward<A21>( a21 ))
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
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19 , BOOST_FWD_REF(A20) a20 , BOOST_FWD_REF(A21) a21 , BOOST_FWD_REF(A22) a22)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 ) , boost::forward<A20>( a20 ) , boost::forward<A21>( a21 ) , boost::forward<A22>( a22 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21 , typename A22 , typename A23>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19 , BOOST_FWD_REF(A20) a20 , BOOST_FWD_REF(A21) a21 , BOOST_FWD_REF(A22) a22 , BOOST_FWD_REF(A23) a23)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 ) , boost::forward<A20>( a20 ) , boost::forward<A21>( a21 ) , boost::forward<A22>( a22 ) , boost::forward<A23>( a23 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21 , typename A22 , typename A23 , typename A24>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19 , BOOST_FWD_REF(A20) a20 , BOOST_FWD_REF(A21) a21 , BOOST_FWD_REF(A22) a22 , BOOST_FWD_REF(A23) a23 , BOOST_FWD_REF(A24) a24)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 ) , boost::forward<A20>( a20 ) , boost::forward<A21>( a21 ) , boost::forward<A22>( a22 ) , boost::forward<A23>( a23 ) , boost::forward<A24>( a24 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21 , typename A22 , typename A23 , typename A24 , typename A25>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19 , BOOST_FWD_REF(A20) a20 , BOOST_FWD_REF(A21) a21 , BOOST_FWD_REF(A22) a22 , BOOST_FWD_REF(A23) a23 , BOOST_FWD_REF(A24) a24 , BOOST_FWD_REF(A25) a25)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 ) , boost::forward<A20>( a20 ) , boost::forward<A21>( a21 ) , boost::forward<A22>( a22 ) , boost::forward<A23>( a23 ) , boost::forward<A24>( a24 ) , boost::forward<A25>( a25 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21 , typename A22 , typename A23 , typename A24 , typename A25 , typename A26>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19 , BOOST_FWD_REF(A20) a20 , BOOST_FWD_REF(A21) a21 , BOOST_FWD_REF(A22) a22 , BOOST_FWD_REF(A23) a23 , BOOST_FWD_REF(A24) a24 , BOOST_FWD_REF(A25) a25 , BOOST_FWD_REF(A26) a26)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 ) , boost::forward<A20>( a20 ) , boost::forward<A21>( a21 ) , boost::forward<A22>( a22 ) , boost::forward<A23>( a23 ) , boost::forward<A24>( a24 ) , boost::forward<A25>( a25 ) , boost::forward<A26>( a26 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21 , typename A22 , typename A23 , typename A24 , typename A25 , typename A26 , typename A27>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19 , BOOST_FWD_REF(A20) a20 , BOOST_FWD_REF(A21) a21 , BOOST_FWD_REF(A22) a22 , BOOST_FWD_REF(A23) a23 , BOOST_FWD_REF(A24) a24 , BOOST_FWD_REF(A25) a25 , BOOST_FWD_REF(A26) a26 , BOOST_FWD_REF(A27) a27)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 ) , boost::forward<A20>( a20 ) , boost::forward<A21>( a21 ) , boost::forward<A22>( a22 ) , boost::forward<A23>( a23 ) , boost::forward<A24>( a24 ) , boost::forward<A25>( a25 ) , boost::forward<A26>( a26 ) , boost::forward<A27>( a27 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21 , typename A22 , typename A23 , typename A24 , typename A25 , typename A26 , typename A27 , typename A28>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19 , BOOST_FWD_REF(A20) a20 , BOOST_FWD_REF(A21) a21 , BOOST_FWD_REF(A22) a22 , BOOST_FWD_REF(A23) a23 , BOOST_FWD_REF(A24) a24 , BOOST_FWD_REF(A25) a25 , BOOST_FWD_REF(A26) a26 , BOOST_FWD_REF(A27) a27 , BOOST_FWD_REF(A28) a28)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 ) , boost::forward<A20>( a20 ) , boost::forward<A21>( a21 ) , boost::forward<A22>( a22 ) , boost::forward<A23>( a23 ) , boost::forward<A24>( a24 ) , boost::forward<A25>( a25 ) , boost::forward<A26>( a26 ) , boost::forward<A27>( a27 ) , boost::forward<A28>( a28 ))
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
    template <typename T, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21 , typename A22 , typename A23 , typename A24 , typename A25 , typename A26 , typename A27 , typename A28 , typename A29>
    std::vector<lcos::future<object<T> > >
    distributed_new(std::size_t count, BOOST_FWD_REF(A0) a0 , BOOST_FWD_REF(A1) a1 , BOOST_FWD_REF(A2) a2 , BOOST_FWD_REF(A3) a3 , BOOST_FWD_REF(A4) a4 , BOOST_FWD_REF(A5) a5 , BOOST_FWD_REF(A6) a6 , BOOST_FWD_REF(A7) a7 , BOOST_FWD_REF(A8) a8 , BOOST_FWD_REF(A9) a9 , BOOST_FWD_REF(A10) a10 , BOOST_FWD_REF(A11) a11 , BOOST_FWD_REF(A12) a12 , BOOST_FWD_REF(A13) a13 , BOOST_FWD_REF(A14) a14 , BOOST_FWD_REF(A15) a15 , BOOST_FWD_REF(A16) a16 , BOOST_FWD_REF(A17) a17 , BOOST_FWD_REF(A18) a18 , BOOST_FWD_REF(A19) a19 , BOOST_FWD_REF(A20) a20 , BOOST_FWD_REF(A21) a21 , BOOST_FWD_REF(A22) a22 , BOOST_FWD_REF(A23) a23 , BOOST_FWD_REF(A24) a24 , BOOST_FWD_REF(A25) a25 , BOOST_FWD_REF(A26) a26 , BOOST_FWD_REF(A27) a27 , BOOST_FWD_REF(A28) a28 , BOOST_FWD_REF(A29) a29)
    {
        hpx::components::component_type type
            = hpx::components::get_component_type<hpx::components::server::remote_object>();
        std::vector<naming::id_type> prefixes = find_all_localities(type);
        std::vector<naming::id_type>::size_type objs_per_loc = count / prefixes.size();
        std::size_t created_count = 0;
        std::size_t excess = count - objs_per_loc*prefixes.size();
        std::vector<lcos::future<object<T> > > res;
        res.reserve(count);
        __pragma(warning(suppress:6001)) if (boost::foreach_detail_::auto_any_t _foreach_col = boost::foreach_detail_::contain( ( prefixes) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_cur = boost::foreach_detail_::begin( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else if (boost::foreach_detail_::auto_any_t _foreach_end = boost::foreach_detail_::end( _foreach_col , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes))) , (true ? 0 : boost::foreach_detail_::or_( boost::foreach_detail_::is_rvalue_(( prefixes), 0) , boost::foreach_detail_::and_( boost::foreach_detail_::not_(boost_foreach_is_noncopyable( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)) , boost_foreach_is_lightweight_proxy( boost::foreach_detail_::to_ptr( prefixes) , boost_foreach_argument_dependent_lookup_hack_value)))))) {} else for (bool _foreach_continue = true; _foreach_continue && !boost::foreach_detail_::done( _foreach_cur , _foreach_end , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); _foreach_continue ? boost::foreach_detail_::next( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))) : (void)0) if (boost::foreach_detail_::set_false(_foreach_continue)) {} else for (naming::id_type const & prefix = boost::foreach_detail_::deref( _foreach_cur , (true ? 0 : boost::foreach_detail_::encode_type( prefixes, boost::foreach_detail_::is_const_( prefixes)))); !_foreach_continue; _foreach_continue = true)
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
                    new_<T>(prefix, boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ) , boost::forward<A8>( a8 ) , boost::forward<A9>( a9 ) , boost::forward<A10>( a10 ) , boost::forward<A11>( a11 ) , boost::forward<A12>( a12 ) , boost::forward<A13>( a13 ) , boost::forward<A14>( a14 ) , boost::forward<A15>( a15 ) , boost::forward<A16>( a16 ) , boost::forward<A17>( a17 ) , boost::forward<A18>( a18 ) , boost::forward<A19>( a19 ) , boost::forward<A20>( a20 ) , boost::forward<A21>( a21 ) , boost::forward<A22>( a22 ) , boost::forward<A23>( a23 ) , boost::forward<A24>( a24 ) , boost::forward<A25>( a25 ) , boost::forward<A26>( a26 ) , boost::forward<A27>( a27 ) , boost::forward<A28>( a28 ) , boost::forward<A29>( a29 ))
                );
            }
            created_count += numcreate;
            if (created_count >= count)
                break;
        }
        return res;
    }
}}
