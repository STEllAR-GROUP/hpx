//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_USE_ITT)

#include <hpx/util/static.hpp>
#include <hpx/util/itt_notify.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/plugin.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/filesystem.hpp>

#include <list>

///////////////////////////////////////////////////////////////////////////////
// This file initializes the stubs which can be used to dynamically load and 
// call the itt-notify functions connecting the HPX synchronization objects
// with the Intel Inspector Tool
namespace hpx { namespace util
{
    namespace detail
    {
        struct init_itt
        {
            typedef void (*function_pointer_type)();
            typedef boost::function<void(function_pointer_type)> deleter_type;
            typedef boost::remove_pointer<function_pointer_type>::type 
                function_type;
            typedef std::map<std::string, boost::shared_ptr<function_type> >
                entry_map_type;

            init_itt() : libitt_(0)
            {
                namespace fs = boost::filesystem;
                try {
                    fs::path p("libittnotify");
                    p.replace_extension(HPX_SHARED_LIB_EXTENSION);
                    libitt_ = new boost::plugin::dll(p.string());
                }
                catch (std::logic_error const&) {
                    libitt_ = 0;
                }
            }

            ~init_itt()
            {
                loaded_entries_.clear();
                delete libitt_;
            }

            template <typename FuncType>
            FuncType load(char const* funcname)
            {
                if (0 == libitt_)
                    return 0;

                try {
                    std::pair<function_pointer_type, deleter_type> p = 
                        libitt_->get<function_pointer_type, deleter_type>(funcname);
                    if (p.first) {
                        loaded_entries_.insert(
                            entry_map_type::value_type(funcname,
                                boost::shared_ptr<function_type>(p.first, p.second)
                            ));
                    }
                    return reinterpret_cast<FuncType>(p.first);
                }
                catch (std::logic_error const&) {
                    return 0;
                }
            }

            bool unload(char const* funcname)
            {
                entry_map_type::iterator it = loaded_entries_.find(funcname);
                if (it == loaded_entries_.end())
                    return false;

                loaded_entries_.erase(it);
                return true;
            }

            boost::plugin::dll* libitt_;
            entry_map_type loaded_entries_;
        };

        init_itt& get_itt_init()
        {
            static_<init_itt> itt_init_;
            return itt_init_.get();
        }
    }

#define HPX_ITT_INIT(init, f)                                                 \
    ITTNOTIFY_NAME(f) = init.load<BOOST_PP_CAT(ITTNOTIFY_NAME(f), _t)>(       \
        BOOST_PP_STRINGIZE_I(__itt_ ## f))                                    \
    /**/
#define HPX_ITT_DEINIT(init, f)                                               \
    init.unload(BOOST_PP_STRINGIZE_I(__itt_ ## f));                           \
    ITTNOTIFY_NAME(f) = 0                                                     \
    /**/

    // initialize stub object to call into the ittnotify library, if present
    void init_itt_api()
    {
        detail::init_itt& init = detail::get_itt_init();

#if defined(BOOST_WINDOWS)
        HPX_ITT_INIT(init, sync_createA);
#else
        HPX_ITT_INIT(init, sync_create);
#endif
        HPX_ITT_INIT(init, sync_prepare);
        HPX_ITT_INIT(init, sync_cancel);
        HPX_ITT_INIT(init, sync_acquired);
        HPX_ITT_INIT(init, sync_releasing);
        HPX_ITT_INIT(init, sync_released);
        HPX_ITT_INIT(init, sync_destroy);
    }

    // de-initialize stub objects
    void deinit_itt_api()
    {
        detail::init_itt& init = detail::get_itt_init();

#if defined(BOOST_WINDOWS)
        HPX_ITT_DEINIT(init, sync_createA);
#else
        HPX_ITT_DEINIT(init, sync_create);
#endif
        HPX_ITT_DEINIT(init, sync_prepare);
        HPX_ITT_DEINIT(init, sync_cancel);
        HPX_ITT_DEINIT(init, sync_acquired);
        HPX_ITT_DEINIT(init, sync_releasing);
        HPX_ITT_DEINIT(init, sync_released);
        HPX_ITT_DEINIT(init, sync_destroy);
    }
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_ITT_DEFINE_STUB(name)                                             \
    BOOST_PP_CAT(ITTNOTIFY_NAME(name), _t) ITTNOTIFY_NAME(name) = 0           \
    /**/

#if defined(BOOST_WINDOWS)
HPX_ITT_DEFINE_STUB(sync_createA);
#else
HPX_ITT_DEFINE_STUB(sync_create);
#endif
HPX_ITT_DEFINE_STUB(sync_prepare);
HPX_ITT_DEFINE_STUB(sync_cancel);
HPX_ITT_DEFINE_STUB(sync_acquired);
HPX_ITT_DEFINE_STUB(sync_releasing);
HPX_ITT_DEFINE_STUB(sync_released);
HPX_ITT_DEFINE_STUB(sync_destroy);

#endif // HPX_USE_ITT
