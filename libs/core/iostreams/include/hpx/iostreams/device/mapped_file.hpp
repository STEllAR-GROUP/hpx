//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright Jorge Lodos 2008.
// (C) Copyright Jonathan Turkanis 2003.
// (C) Copyright Craig Henderson 2002.   'boost/memmap.hpp' from sandbox

#pragma once

#include <hpx/config.hpp>
#include <hpx/iostreams/config/defines.hpp>
#include <hpx/iostreams/close.hpp>
#include <hpx/iostreams/concepts.hpp>
#include <hpx/iostreams/operations_fwd.hpp>
#include <hpx/iostreams/positioning.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx::iostreams {

    //------------------Definition of mapped_file_base and mapped_file_params-----//

    // Forward declarations
    class mapped_file_source;
    class mapped_file_sink;
    class mapped_file;

    namespace detail {

        class mapped_file_impl;
    }    // namespace detail

    class mapped_file_base
    {
    public:
        enum class mapmode : std::uint8_t
        {
            none = 0,
            readonly = 1,
            readwrite = 2,
            priv = 4
        };
    };

    // Bitmask operations for mapped_file_base::mapmode
    constexpr mapped_file_base::mapmode operator|(
        mapped_file_base::mapmode a, mapped_file_base::mapmode b) noexcept;

    constexpr mapped_file_base::mapmode operator&(
        mapped_file_base::mapmode a, mapped_file_base::mapmode b) noexcept;

    constexpr mapped_file_base::mapmode operator^(
        mapped_file_base::mapmode a, mapped_file_base::mapmode b) noexcept;

    constexpr mapped_file_base::mapmode operator~(
        mapped_file_base::mapmode a) noexcept;

    constexpr mapped_file_base::mapmode operator|=(
        mapped_file_base::mapmode& a, mapped_file_base::mapmode b) noexcept;

    constexpr mapped_file_base::mapmode operator&=(
        mapped_file_base::mapmode& a, mapped_file_base::mapmode b) noexcept;

    constexpr mapped_file_base::mapmode operator^=(
        mapped_file_base::mapmode& a, mapped_file_base::mapmode b) noexcept;

    //------------------Definition of mapped_file_params--------------------------//
    namespace detail {

        struct mapped_file_params_base
        {
            mapped_file_params_base()
              : flags(static_cast<mapped_file_base::mapmode>(0))
              , mode()
              , offset(0)
              , length(static_cast<std::size_t>(-1))
              , new_file_size(0)
              , hint(0)
            {
            }

        private:
            friend class mapped_file_impl;
            void normalize();

        public:
            mapped_file_base::mapmode flags;
            std::ios_base::openmode mode;    // Deprecated
            stream_offset offset;
            std::size_t length;
            stream_offset new_file_size;
            char const* hint;
        };
    }    // namespace detail

    // This template allows Std.Filesystem paths to be specified when creating
    // or reopening a memory mapped file, without creating a dependence on
    // Std.Filesystem. Possible values of Path include std::string,
    // std::filesystem::path, std::filesystem::wpath, and
    // hpx::iostreams::detail::path (used to store either a std::string or a
    // std::wstring).
    template <typename Path>
    struct basic_mapped_file_params : detail::mapped_file_params_base
    {
        typedef detail::mapped_file_params_base base_type;

#if defined(HPX_IOSTREAMS_HAVE_WIDE_STREAMS)
        // For wide paths, instantiate basic_mapped_file_params
        // with std::filesystem::wpath
        static_assert(!std::is_same_v<Path, std::wstring>));
#endif

        // Default constructor
        basic_mapped_file_params() = default;

        // Construction from a Path
        explicit basic_mapped_file_params(Path const& p)
          : path(p)
        {
        }

        // Construction from a path of a different type
        template <typename PathT>
        explicit basic_mapped_file_params(PathT const& p)
          : path(p)
        {
        }

        // Copy constructor
        basic_mapped_file_params(basic_mapped_file_params const& other)
          : base_type(other)
          , path(other.path)
        {
        }

        // Templated copy constructor
        template <typename PathT>
        basic_mapped_file_params(basic_mapped_file_params<PathT> const& other)
          : base_type(other)
          , path(other.path)
        {
        }

        using path_type = Path;
        Path path;
    };

    using mapped_file_params = basic_mapped_file_params<std::string>;

    //------------------Definition of mapped_file_source--------------------------//
    class HPX_CORE_EXPORT mapped_file_source : public mapped_file_base
    {
    private:
        typedef detail::mapped_file_impl impl_type;
        typedef basic_mapped_file_params<std::filesystem::path> param_type;

        friend class mapped_file;
        friend class detail::mapped_file_impl;
        friend struct hpx::iostreams::operations<mapped_file_source>;

    public:
        typedef char char_type;

        struct category
          : public source_tag
          , public direct_tag
          , public closable_tag
        {
        };

        typedef std::size_t size_type;
        typedef char const* iterator;

        static constexpr size_type max_length = static_cast<size_type>(-1);

        // Default constructor
        mapped_file_source();

        // Constructor taking a parameters object
        template <typename Path>
        explicit mapped_file_source(basic_mapped_file_params<Path> const& p);

        // Constructor taking a list of parameters
        template <typename Path>
        explicit mapped_file_source(Path const& path,
            size_type length = max_length, boost::intmax_t offset = 0);

        // Copy Constructor
        mapped_file_source(mapped_file_source const& other);

        //--------------Stream interface------------------------------------------//
        template <typename Path>
        void open(basic_mapped_file_params<Path> const& p);

        template <typename Path>
        void open(Path const& path, size_type length = max_length,
            boost::intmax_t offset = 0);

        bool is_open() const;
        void close();

        explicit operator bool() const;
        bool operator!() const;

        mapmode flags() const;

        //--------------Container interface---------------------------------------//
        char const* data() const;
        size_type size() const;

        iterator begin() const;
        iterator end() const;

        //--------------Query admissible offsets----------------------------------//

        // Returns the allocation granularity for virtual memory. Values passed
        // as offsets must be multiples of this value.
        static int alignment();

    private:
        void init();
        void open_impl(param_type const& p);

        std::shared_ptr<impl_type> pimpl_;
    };

    //------------------Definition of mapped_file---------------------------------//
    class HPX_CORE_EXPORT mapped_file : public mapped_file_base
    {
    private:
        typedef mapped_file_source delegate_type;
        typedef basic_mapped_file_params<std::filesystem::path> param_type;

        friend struct hpx::iostreams::operations<mapped_file>;
        friend class mapped_file_sink;

    public:
        typedef char char_type;

        struct category
          : public seekable_device_tag
          , public direct_tag
          , public closable_tag
        {
        };

        typedef mapped_file_source::size_type size_type;
        typedef char* iterator;
        typedef char const* const_iterator;

        static constexpr size_type max_length = delegate_type::max_length;

        // Default constructor
        mapped_file() = default;

        // Construstor taking a parameters object
        template <typename Path>
        explicit mapped_file(basic_mapped_file_params<Path> const& p);

        // Constructor taking a list of parameters
        template <typename Path>
        mapped_file(Path const& path, mapmode flags,
            size_type length = max_length, stream_offset offset = 0);

        // Constructor taking a list of parameters, including a
        // std::ios_base::openmode (deprecated)
        template <typename Path>
        explicit mapped_file(Path const& path,
            std::ios_base::openmode mode = std::ios_base::in |
                std::ios_base::out,
            size_type length = max_length, stream_offset offset = 0);

        // Copy Constructor
        mapped_file(mapped_file const& other);

        //--------------Conversion to mapped_file_source (deprecated)-------------//
        operator mapped_file_source&()
        {
            return delegate_;
        }
        operator mapped_file_source const&() const
        {
            return delegate_;
        }

        //--------------Stream interface------------------------------------------//

        // open overload taking a parameters object
        template <typename Path>
        void open(basic_mapped_file_params<Path> const& p);

        // open overload taking a list of parameters
        template <typename Path>
        void open(Path const& path, mapmode mode, size_type length = max_length,
            stream_offset offset = 0);

        // open overload taking a list of parameters, including a
        // std::ios_base::openmode (deprecated)
        template <typename Path>
        void open(Path const& path,
            std::ios_base::openmode mode = std::ios_base::in |
                std::ios_base::out,
            size_type length = max_length, stream_offset offset = 0);

        bool is_open() const
        {
            return delegate_.is_open();
        }

        void close()
        {
            delegate_.close();
        }

        explicit operator bool() const
        {
            return bool(delegate_);
        }
        bool operator!() const
        {
            return !bool(delegate_);
        }

        mapmode flags() const
        {
            return delegate_.flags();
        }

        //--------------Container interface---------------------------------------//
        size_type size() const
        {
            return delegate_.size();
        }

        char* data() const;
        char const* const_data() const
        {
            return delegate_.data();
        }

        iterator begin() const
        {
            return data();
        }
        const_iterator const_begin() const
        {
            return const_data();
        }

        iterator end() const;
        const_iterator const_end() const
        {
            return const_data() + size();
        }

        //--------------Query admissible offsets----------------------------------//

        // Returns the allocation granularity for virtual memory. Values passed
        // as offsets must be multiples of this value.
        static int alignment()
        {
            return mapped_file_source::alignment();
        }

        //--------------File access----------------------------------------------//
        void resize(stream_offset new_size);

    private:
        delegate_type delegate_;
    };

    //------------------Definition of mapped_file_sink----------------------------//
    class HPX_CORE_EXPORT mapped_file_sink : private mapped_file
    {
    public:
        friend struct hpx::iostreams::operations<mapped_file_sink>;

        using mapped_file::char_type;
        using mapped_file::mapmode;

        struct category
          : public sink_tag
          , public direct_tag
          , public closable_tag
        {
        };

        using mapped_file::close;
        using mapped_file::is_open;
        using mapped_file::iterator;
        using mapped_file::max_length;
        using mapped_file::size_type;
        using mapped_file::operator bool;
        using mapped_file::operator!;
        using mapped_file::alignment;
        using mapped_file::begin;
        using mapped_file::data;
        using mapped_file::end;
        using mapped_file::flags;
        using mapped_file::resize;
        using mapped_file::size;

        // Default constructor
        mapped_file_sink() = default;

        // Constructor taking a parameters object
        template <typename Path>
        explicit mapped_file_sink(basic_mapped_file_params<Path> const& p);

        // Constructor taking a list of parameters
        template <typename Path>
        explicit mapped_file_sink(Path const& path,
            size_type length = max_length, boost::intmax_t offset = 0,
            mapmode flags = mapmode::readwrite);

        // Copy Constructor
        mapped_file_sink(mapped_file_sink const& other);

        // open overload taking a parameters object
        template <typename Path>
        void open(basic_mapped_file_params<Path> const& p);

        // open overload taking a list of parameters
        template <typename Path>
        void open(Path const& path, size_type length = max_length,
            boost::intmax_t offset = 0, mapmode flags = readwrite);
    };

    //------------------Implementation of mapped_file_source----------------------//
    template <typename Path>
    mapped_file_source::mapped_file_source(
        basic_mapped_file_params<Path> const& p)
    {
        init();
        open(p);
    }

    template <typename Path>
    mapped_file_source::mapped_file_source(
        Path const& path, size_type length, boost::intmax_t offset)
    {
        init();
        open(path, length, offset);
    }

    template <typename Path>
    void mapped_file_source::open(basic_mapped_file_params<Path> const& p)
    {
        param_type params(p);
        if (params.flags != mapped_file::mapmode::none)
        {
            if (params.flags != mapped_file::mapmode::readonly)
            {
                throw std::ios_base::failure("invalid flags");
            }
        }
        else
        {
            if (params.mode & std::ios_base::out)
            {
                throw std::ios_base::failure("invalid mode");
            }

            params.mode |= std::ios_base::in;
        }
        open_impl(params);
    }

    template <typename Path>
    void mapped_file_source::open(
        Path const& path, size_type length, boost::intmax_t offset)
    {
        param_type p(path);
        p.length = length;
        p.offset = offset;
        open(p);
    }

    //------------------Implementation of mapped_file-----------------------------//
    template <typename Path>
    mapped_file::mapped_file(basic_mapped_file_params<Path> const& p)
    {
        open(p);
    }

    template <typename Path>
    mapped_file::mapped_file(
        Path const& path, mapmode flags, size_type length, stream_offset offset)
    {
        open(path, flags, length, offset);
    }

    template <typename Path>
    mapped_file::mapped_file(Path const& path, std::ios_base::openmode mode,
        size_type length, stream_offset offset)
    {
        open(path, mode, length, offset);
    }

    template <typename Path>
    void mapped_file::open(basic_mapped_file_params<Path> const& p)
    {
        delegate_.open_impl(p);
    }

    template <typename Path>
    void mapped_file::open(
        Path const& path, mapmode flags, size_type length, stream_offset offset)
    {
        param_type p(path);
        p.flags = flags;
        p.length = length;
        p.offset = offset;
        open(p);
    }

    template <typename Path>
    void mapped_file::open(Path const& path, std::ios_base::openmode mode,
        size_type length, stream_offset offset)
    {
        param_type p(path);
        p.mode = mode;
        p.length = length;
        p.offset = offset;
        open(p);
    }

    inline char* mapped_file::data() const
    {
        return (flags() != mapmode::readonly) ?
            const_cast<char*>(delegate_.data()) :
            0;
    }

    inline mapped_file::iterator mapped_file::end() const
    {
        return (flags() != mapmode::readonly) ? data() + size() : 0;
    }

    //------------------Implementation of mapped_file_sink------------------------//
    template <typename Path>
    mapped_file_sink::mapped_file_sink(basic_mapped_file_params<Path> const& p)
    {
        open(p);
    }

    template <typename Path>
    mapped_file_sink::mapped_file_sink(Path const& path, size_type length,
        boost::intmax_t offset, mapmode flags)
    {
        open(path, length, offset, flags);
    }

    template <typename Path>
    void mapped_file_sink::open(basic_mapped_file_params<Path> const& p)
    {
        param_type params(p);
        if (params.flags != mapped_file::mapmode::none)
        {
            if ((params.flags & mapped_file::mapmode::readonly) !=
                mapped_file::mapmode::none)
            {
                throw std::ios_base::failure("invalid flags");
            }
        }
        else
        {
            if (params.mode & std::ios_base::in)
            {
                throw std::ios_base::failure("invalid mode");
            }
            params.mode |= std::ios_base::out;
        }

        mapped_file::open(params);
    }

    template <typename Path>
    void mapped_file_sink::open(
        Path const& path, size_type length, std::intmax_t offset, mapmode flags)
    {
        param_type p(path);
        p.flags = flags;
        p.length = length;
        p.offset = offset;
        open(p);
    }

    //------------------Specialization of direct_impl-----------------------------//
    template <>
    struct operations<mapped_file_source> : detail::close_impl<closable_tag>
    {
        static std::span<char> input_sequence(mapped_file_source const& src)
        {
            return {
                const_cast<char*>(src.begin()), const_cast<char*>(src.end())};
        }
    };

    template <>
    struct operations<mapped_file> : detail::close_impl<closable_tag>
    {
        static std::span<char> input_sequence(mapped_file const& file)
        {
            return {file.begin(), file.end()};
        }
        static std::span<char> output_sequence(mapped_file const& file)
        {
            return {file.begin(), file.end()};
        }
    };

    template <>
    struct operations<mapped_file_sink> : detail::close_impl<closable_tag>
    {
        static std::span<char> output_sequence(mapped_file_sink const& sink)
        {
            return {sink.begin(), sink.end()};
        }
    };

    //------------------Definition of mapmode operators---------------------------//
    constexpr mapped_file::mapmode operator|(
        mapped_file::mapmode const a, mapped_file::mapmode const b) noexcept
    {
        return static_cast<mapped_file::mapmode>(
            static_cast<int>(a) | static_cast<int>(b));
    }

    constexpr mapped_file::mapmode operator&(
        mapped_file::mapmode const a, mapped_file::mapmode const b) noexcept
    {
        return static_cast<mapped_file::mapmode>(
            static_cast<int>(a) & static_cast<int>(b));
    }

    constexpr mapped_file::mapmode operator^(
        mapped_file::mapmode const a, mapped_file::mapmode const b) noexcept
    {
        return static_cast<mapped_file::mapmode>(
            static_cast<int>(a) ^ static_cast<int>(b));
    }

    constexpr mapped_file::mapmode operator~(
        mapped_file::mapmode const a) noexcept
    {
        return static_cast<mapped_file::mapmode>(~static_cast<int>(a));
    }

    constexpr mapped_file::mapmode operator|=(
        mapped_file::mapmode& a, mapped_file::mapmode const b) noexcept
    {
        return a = a | b;
    }

    constexpr mapped_file::mapmode operator&=(
        mapped_file::mapmode& a, mapped_file::mapmode const b) noexcept
    {
        return a = a & b;
    }

    constexpr mapped_file::mapmode operator^=(
        mapped_file::mapmode& a, mapped_file::mapmode const b) noexcept
    {
        return a = a ^ b;
    }
}    // namespace hpx::iostreams
