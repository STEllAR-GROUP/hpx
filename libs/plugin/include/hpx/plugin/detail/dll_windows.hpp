// Copyright Vladimir Prus 2004.
// Copyright (c) 2005-2012 Hartmut Kaiser
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/plugin/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/filesystem.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <Shlwapi.h>
#include <windows.h>

#if !defined(HPX_MSVC) && !defined(HPX_MINGW)
#error                                                                         \
    "This file shouldn't be included directly, use the file hpx/plugin/dll.hpp only."
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace plugin {

    namespace detail {

        template <typename T>
        struct free_dll
        {
            free_dll(HMODULE h)
              : h(h)
            {
            }

            void operator()(T)
            {
                if (nullptr != h)
                    FreeLibrary(h);
            }

            HMODULE h;
        };
    }    // namespace detail

    class dll
    {
    public:
        dll()
          : dll_handle(nullptr)
        {
        }

        dll(dll const& rhs)
          : dll_name(rhs.dll_name)
          , map_name(rhs.map_name)
          , dll_handle(nullptr)
        {
        }

        dll(std::string const& libname)
          : dll_name(libname)
          , map_name()
          , dll_handle(nullptr)
        {
            // map_name defaults to dll base name
            namespace fs = filesystem;

            fs::path dll_path(dll_name);
            map_name = fs::basename(dll_path);
        }

        void load_library(error_code& ec = throws)
        {
            LoadLibrary(ec);
        }

        dll(std::string const& libname, std::string const& mapname)
          : dll_name(libname)
          , map_name(mapname)
          , dll_handle(nullptr)
        {
        }

        dll(dll&& rhs) noexcept
          : dll_name(std::move(rhs.dll_name))
          , map_name(std::move(rhs.map_name))
          , dll_handle(rhs.dll_handle)
        {
            rhs.dll_handle = nullptr;
        }

        dll& operator=(dll const& rhs)
        {
            if (this != &rhs)
            {
                //  free any existing dll_handle
                FreeLibrary();

                //  load the library for this instance of the dll class
                dll_name = rhs.dll_name;
                map_name = rhs.map_name;
                LoadLibrary();
            }
            return *this;
        }

        dll& operator=(dll&& rhs) noexcept
        {
            if (&rhs != this)
            {
                dll_name = std::move(rhs.dll_name);
                map_name = std::move(rhs.map_name);
                dll_handle = rhs.dll_handle;
                rhs.dll_handle = nullptr;
            }
            return *this;
        }

        ~dll()
        {
            FreeLibrary();
        }

        std::string const& get_name() const
        {
            return dll_name;
        }
        std::string const& get_mapname() const
        {
            return map_name;
        }

        template <typename SymbolType, typename Deleter>
        std::pair<SymbolType, Deleter> get(
            std::string const& symbol_name, error_code& ec = throws) const
        {
            const_cast<dll&>(*this).LoadLibrary(ec);
            // make sure everything is initialized
            if (ec)
                return std::pair<SymbolType, Deleter>();

            static_assert(std::is_pointer<SymbolType>::value,
                "std::is_pointer<SymbolType>::value");

            // Cast the to right type.
            SymbolType address =
                (SymbolType) GetProcAddress(dll_handle, symbol_name.c_str());
            if (nullptr == address)
            {
                std::ostringstream str;
                str << "Hpx.Plugin: Unable to locate the exported symbol name '"
                    << symbol_name << "' in the shared library '" << dll_name
                    << "'";

                // report error
                HPX_THROWS_IF(
                    ec, dynamic_link_failure, "plugin::get", str.str());
                return std::pair<SymbolType, Deleter>();
            }

            // Open the library. Yes, we do it on every access to
            // a symbol, the LoadLibrary function increases the refcnt of the dll
            // so in the end the dll class holds one refcnt and so does every
            // symbol.
            HMODULE handle = ::LoadLibraryA(dll_name.c_str());
            if (!handle)
            {
                std::ostringstream str;
                str << "Hpx.Plugin: Could not open shared library '" << dll_name
                    << "'";

                // report error
                HPX_THROWS_IF(ec, filesystem_error, "plugin::get", str.str());
                return std::pair<SymbolType, Deleter>();
            }
            HPX_ASSERT(handle == dll_handle);

            return std::make_pair(
                address, detail::free_dll<SymbolType>(handle));
        }

        void keep_alive(error_code& ec = throws)
        {
            LoadLibrary(ec, true);
        }

    protected:
        void LoadLibrary(error_code& ec = throws, bool force = false)
        {
            if (!dll_handle || force)
            {
                if (dll_name.empty())
                {
                    // load main module
                    char buffer[_MAX_PATH];
                    ::GetModuleFileNameA(nullptr, buffer, sizeof(buffer));
                    dll_name = buffer;
                }

                dll_handle = ::LoadLibraryA(dll_name.c_str());
                if (!dll_handle)
                {
                    std::ostringstream str;
                    str << "Hpx.Plugin: Could not open shared library '"
                        << dll_name << "'";

                    HPX_THROWS_IF(
                        ec, filesystem_error, "plugin::LoadLibrary", str.str());
                    return;
                }
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

    public:
        std::string get_directory(error_code& ec = throws) const
        {
            char buffer[_MAX_PATH] = {'\0'};

            const_cast<dll&>(*this).LoadLibrary(ec);
            // make sure everything is initialized
            if (ec)
                return buffer;

            DWORD name_length =
                GetModuleFileNameA(dll_handle, buffer, sizeof(buffer));

            if (name_length <= 0)
            {
                std::ostringstream str;
                str << "Hpx.Plugin: Could not extract path the shared "
                       "library '"
                    << dll_name << "' has been loaded from.";

                HPX_THROWS_IF(
                    ec, filesystem_error, "plugin::get_directory", str.str());
                return buffer;
            }

            // extract the directory name
            PathRemoveFileSpecA(buffer);

            if (&ec != &throws)
                ec = make_success_code();

            return buffer;
        }

    protected:
        void FreeLibrary()
        {
            if (nullptr != dll_handle)
                ::FreeLibrary(dll_handle);
        }

    private:
        std::string dll_name;
        std::string map_name;
        HMODULE dll_handle;
    };

}}}    // namespace hpx::util::plugin
