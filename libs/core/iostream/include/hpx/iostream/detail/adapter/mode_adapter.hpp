//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#pragma once

// Contains the definition of the class template mode_adapter, which allows
// a filter or device to function as if it has a different i/o mode than that
// deduced by the metafunction mode_of.

#include <hpx/config.hpp>

#include <hpx/iostream/categories.hpp>
#include <hpx/iostream/detail/wrap_unwrap.hpp>
#include <hpx/iostream/operations.hpp>
#include <hpx/iostream/traits.hpp>

#include <ios>
#include <type_traits>

namespace hpx::iostream::detail {

    HPX_CXX_CORE_EXPORT template <typename Mode, typename T>
    class mode_adapter
    {
        struct empty_base1
        {
        };

        struct empty_base2
        {
        };

    public:
        using component_type = wrapped_type<T>::type;
        using char_type = char_type_of_t<T>;

        struct category
          : Mode
          , device_tag
          , std::conditional_t<is_filter_v<T>, filter_tag, empty_base1>
          , std::conditional_t<is_filter_v<T>, multichar_tag, empty_base2>
          , closable_tag
          , localizable_tag
        {
        };

        explicit mode_adapter(component_type const& t)
          : t_(t)
        {
        }

        explicit mode_adapter(component_type&& t) noexcept
          : t_(HPX_MOVE(t))
        {
        }

        // Device member functions.
        std::streamsize read(char_type* s, std::streamsize const n)
        {
            return iostream::read(t_, s, n);
        }

        std::streamsize write(char_type const* s, std::streamsize const n)
        {
            return iostream::write(t_, s, n);
        }

        std::streampos seek(stream_offset const off,
            std::ios_base::seekdir const way,
            std::ios_base::openmode const which = std::ios_base::in |
                std::ios_base::out)
        {
            return iostream::seek(t_, off, way, which);
        }

        void close()
        {
            detail::close_all(t_);
        }

        void close(std::ios_base::openmode const which)
        {
            iostream::close(t_, which);
        }

        // Filter member functions.
        template <typename Source>
        std::streamsize read(Source& src, char_type* s, std::streamsize const n)
        {
            return iostream::read(t_, src, s, n);
        }

        template <typename Sink>
        std::streamsize write(
            Sink& snk, char_type const* s, std::streamsize const n)
        {
            return iostream::write(t_, snk, s, n);
        }

        template <typename Device>
        std::streampos seek(Device& dev, stream_offset const off,
            std::ios_base::seekdir const way)
        {
            return iostream::seek(t_, dev, off, way);
        }

        template <typename Device>
        std::streampos seek(Device& dev, stream_offset const off,
            std::ios_base::seekdir const way,
            std::ios_base::openmode const which)
        {
            return iostream::seek(t_, dev, off, way, which);
        }

        template <typename Device>
        void close(Device& dev)
        {
            detail::close_all(t_, dev);
        }

        template <typename Device>
        void close(Device& dev, std::ios_base::openmode const which)
        {
            iostream::close(t_, dev, which);
        }

        template <typename Locale>
        void imbue(Locale const& loc)
        {
            iostream::imbue(t_, loc);
        }

    private:
        component_type t_;
    };
}    // namespace hpx::iostream::detail
