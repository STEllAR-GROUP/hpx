// Copyright Vladimir Prus 2004.
// Copyright (c) 2005-2012 Hartmut Kaiser
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_DLL_WINDOWS_HPP_HK_2005_11_06
#define HPX_DLL_WINDOWS_HPP_HK_2005_11_06

#include <string>
#include <stdexcept>
#include <iostream>

#include <boost/config.hpp>
#include <boost/static_assert.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/throw_exception.hpp>

#include <hpx/util/plugin/config.hpp>

#include <windows.h>
#include <Shlwapi.h>

#if !defined(BOOST_WINDOWS)
#error "This file shouldn't be included directly, use the file hpx/util/plugin/dll.hpp only."
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace plugin {

    namespace detail
    {
        template<typename T>
        struct free_dll
        {
            free_dll(HMODULE h) : h(h) {}

            void operator()(T)
            {
                if (NULL != h)
                    FreeLibrary(h);
            }

            HMODULE h;
        };
    }

    class dll
    {
    public:
        dll()
        :   dll_handle(NULL)
        {
            LoadLibrary();
        }

        dll(dll const& rhs)
        :   dll_name(rhs.dll_name), map_name(rhs.map_name), dll_handle(NULL)
        {
            LoadLibrary();
        }

        dll(std::string const& libname)
        :   dll_name(libname), map_name(""), dll_handle(NULL)
        {
            // map_name defaults to dll base name
            namespace fs = boost::filesystem;

#if BOOST_FILESYSTEM_VERSION == 2
            fs::path dll_path(dll_name, fs::native);
#else
            fs::path dll_path(dll_name);
#endif
            map_name = fs::basename(dll_path);

            LoadLibrary();
        }

        dll(std::string const& libname, std::string const& mapname)
        :   dll_name(libname), map_name(mapname), dll_handle(NULL)
        {
            LoadLibrary();
        }

        dll &operator=(dll const& rhs)
        {
            if (this != &rhs) {
            //  free any existing dll_handle
                FreeLibrary();

            //  load the library for this instance of the dll class
                dll_name = rhs.dll_name;
                map_name = rhs.map_name;
                LoadLibrary();
            }
            return *this;
        }

        ~dll()
        {
            FreeLibrary();
        }

        std::string get_name() const { return dll_name; }
        std::string get_mapname() const { return map_name; }

        template<typename SymbolType, typename Deleter>
        std::pair<SymbolType, Deleter>
        get(std::string const& symbol_name) const
        {
            BOOST_STATIC_ASSERT(boost::is_pointer<SymbolType>::value);
            typedef typename boost::remove_pointer<SymbolType>::type PointedType;

            // Open the library. Yes, we do it on every access to
            // a symbol, the LoadLibrary function increases the refcnt of the dll
            // so in the end the dll class holds one refcnt and so does every
            // symbol.
            HMODULE handle = ::LoadLibrary(dll_name.c_str());
            if (!handle) {
                HPX_PLUGIN_OSSTREAM str;
                str << "Hpx.Plugin: Could not open shared library '"
                    << dll_name << "'";

                boost::throw_exception(
                    std::logic_error(HPX_PLUGIN_OSSTREAM_GETSTRING(str)));
            }
            BOOST_ASSERT(handle == dll_handle);

            // Cast the to right type.
            SymbolType address = (SymbolType)GetProcAddress(dll_handle, symbol_name.c_str());
            if (NULL == address)
            {
                HPX_PLUGIN_OSSTREAM str;
                str << "Hpx.Plugin: Unable to locate the exported symbol name '"
                    << symbol_name << "' in the shared library '"
                    << dll_name << "'";

                ::FreeLibrary(handle);
                boost::throw_exception(
                    std::logic_error(HPX_PLUGIN_OSSTREAM_GETSTRING(str)));
            }
            return std::make_pair(address, detail::free_dll<SymbolType>(handle));
        }

        void keep_alive()
        {
            LoadLibrary();
        }

    protected:
        void LoadLibrary()
        {
            if (dll_name.empty()) {
            // load main module
                char buffer[_MAX_PATH];
                ::GetModuleFileName(NULL, buffer, sizeof(buffer));
                dll_name = buffer;
            }

            dll_handle = ::LoadLibrary(dll_name.c_str());
            if (!dll_handle) {
                HPX_PLUGIN_OSSTREAM str;
                str << "Hpx.Plugin: Could not open shared library '"
                    << dll_name << "'";

                boost::throw_exception(
                    std::logic_error(HPX_PLUGIN_OSSTREAM_GETSTRING(str)));
            }
        }

    public:
        std::string get_directory() const
        {
            char buffer[_MAX_PATH];
            DWORD name_length =
                GetModuleFileName(dll_handle, buffer, sizeof(buffer));
            if (name_length <= 0) {
                HPX_PLUGIN_OSSTREAM str;
                str << "Hpx.Plugin: Could not extract path the shared "
                       "library '" << dll_name << "' has been loaded from.";
                boost::throw_exception(
                    std::logic_error(HPX_PLUGIN_OSSTREAM_GETSTRING(str)));
            }

            // extract the directory name
            PathRemoveFileSpec(buffer);
            return buffer;
        }

    protected:
        void FreeLibrary()
        {
            if (NULL != dll_handle)
                ::FreeLibrary(dll_handle);
        }

    private:
        std::string dll_name;
        std::string map_name;
        HMODULE dll_handle;
    };

///////////////////////////////////////////////////////////////////////////////
}}}

#endif

