//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//  See http://www.boost.org/libs/iostreams for documentation.
//
//  Tests the boolean type traits defined in boost/iostreams/traits.hpp.
//
//  File:        libs/iostreams/test/bool_trait_test.cpp
//  Date:        Sun Feb 17 17:52:59 MST 2008
//  Copyright:   2008 CodeRage, LLC
//  Author:      Jonathan Turkanis
//  Contact:     turkanis at coderage dot com

#include <hpx/hpx_main.hpp>
#include <hpx/modules/iostreams.hpp>
#include <hpx/modules/testing.hpp>

#include <fstream>
#include <iostream>
#include <sstream>

using namespace hpx::iostreams;
namespace io = hpx::iostreams;

typedef stream<array_source<char>> array_istream;
typedef stream<array_sink<char>> array_ostream;
typedef stream<array<char>> array_stream;
typedef stream_buffer<array<char>> array_streambuf;

#if defined(HPX_IOSTREAMS_HAVE_WIDE_STREAMS)
typedef stream<warray_source<wchar_t>> array_wistream;
typedef stream<warray_sink<wchar_t>> array_wostream;
typedef stream<warray<wchar_t>> array_wstream;
typedef stream_buffer<warray<wchar_t>> array_wstreambuf;
#endif

typedef io::filtering_stream<seekable> filtering_iostream;
typedef io::detail::linked_streambuf<char> linkedbuf;

#if defined(HPX_IOSTREAMS_HAVE_WIDE_STREAMS)
typedef io::filtering_stream<seekable, wchar_t> filtering_wiostream;
typedef io::detail::linked_streambuf<wchar_t> wlinkedbuf;
#endif

template <template <class> typename Trait, typename Type>
void check_bool_trait(bool status)
{
    HPX_TEST(Trait<Type>::value == status);
}

template <template <class> typename Trait>
void check_stream_trait(bool istream_, [[maybe_unused]] bool wistream_,
    bool ostream_, [[maybe_unused]] bool wostream_, bool iostream_,
    [[maybe_unused]] bool wiostream_, bool streambuf_,
    [[maybe_unused]] bool wstreambuf_, bool ifstream_,
    [[maybe_unused]] bool wifstream_, bool ofstream_,
    [[maybe_unused]] bool wofstream_, bool fstream_,
    [[maybe_unused]] bool wfstream_, bool filebuf_,
    [[maybe_unused]] bool wfilebuf_, bool istringstream_,
    [[maybe_unused]] bool wistringstream_, bool ostringstream_,
    [[maybe_unused]] bool wostringstream_, bool stringstream_,
    [[maybe_unused]] bool wstringstream_, bool stringbuf_,
    [[maybe_unused]] bool wstringbuf_, bool array_istream_,
    [[maybe_unused]] bool array_wistream_, bool array_ostream_,
    [[maybe_unused]] bool array_wostream_, bool array_stream_,
    [[maybe_unused]] bool array_wstream_, bool array_streambuf_,
    [[maybe_unused]] bool array_wstreambuf_, bool filtering_istream_,
    [[maybe_unused]] bool filtering_wistream_, bool filtering_ostream_,
    [[maybe_unused]] bool filtering_wostream_, bool filtering_iostream_,
    [[maybe_unused]] bool filtering_wiostream_, bool filtering_istreambuf_,
    [[maybe_unused]] bool filtering_wistreambuf_, bool linkedbuf_,
    [[maybe_unused]] bool wlinkedbuf_)
{
    check_bool_trait<Trait, std::istream>(istream_);
    check_bool_trait<Trait, std::ostream>(ostream_);
    check_bool_trait<Trait, std::iostream>(iostream_);
    check_bool_trait<Trait, std::streambuf>(streambuf_);
    check_bool_trait<Trait, std::ofstream>(ofstream_);
    check_bool_trait<Trait, std::fstream>(fstream_);
    check_bool_trait<Trait, std::filebuf>(filebuf_);
    check_bool_trait<Trait, std::istringstream>(istringstream_);
    check_bool_trait<Trait, std::ostringstream>(ostringstream_);
    check_bool_trait<Trait, std::stringstream>(stringstream_);
    check_bool_trait<Trait, std::stringbuf>(stringbuf_);

#if defined(HPX_IOSTREAMS_HAVE_WIDE_STREAMS)
    check_bool_trait<Trait, std::wistream>(wistream_);
    check_bool_trait<Trait, std::wostream>(wostream_);
    check_bool_trait<Trait, std::wiostream>(wiostream_);
    check_bool_trait<Trait, std::wstreambuf>(wstreambuf_);
    check_bool_trait<Trait, std::wifstream>(wifstream_);
    check_bool_trait<Trait, std::wofstream>(wofstream_);
    check_bool_trait<Trait, std::wfstream>(wfstream_);
    check_bool_trait<Trait, std::wfilebuf>(wfilebuf_);
    check_bool_trait<Trait, std::wistringstream>(wistringstream_);
    check_bool_trait<Trait, std::wostringstream>(wostringstream_);
    check_bool_trait<Trait, std::wstringstream>(wstringstream_);
    check_bool_trait<Trait, std::wstringbuf>(wstringbuf_);
#endif

    check_bool_trait<Trait, array_istream>(array_istream_);
    check_bool_trait<Trait, array_ostream>(array_ostream_);
    check_bool_trait<Trait, array_stream>(array_stream_);
    check_bool_trait<Trait, array_streambuf>(array_streambuf_);
    check_bool_trait<Trait, filtering_istream>(filtering_istream_);
    check_bool_trait<Trait, filtering_ostream>(filtering_ostream_);
    check_bool_trait<Trait, filtering_iostream>(filtering_iostream_);
    check_bool_trait<Trait, filtering_istreambuf>(filtering_istreambuf_);
    check_bool_trait<Trait, linkedbuf>(linkedbuf_);

#if defined(HPX_IOSTREAMS_HAVE_WIDE_STREAMS)
    check_bool_trait<Trait, array_wistream>(array_wistream_);
    check_bool_trait<Trait, array_wostream>(array_wostream_);
    check_bool_trait<Trait, array_wstream>(array_wstream_);
    check_bool_trait<Trait, array_wstreambuf>(array_wstreambuf_);
    check_bool_trait<Trait, filtering_wistream>(filtering_wistream_);
    check_bool_trait<Trait, filtering_wostream>(filtering_wostream_);
    check_bool_trait<Trait, filtering_wiostream>(filtering_wiostream_);
    check_bool_trait<Trait, filtering_wistreambuf>(filtering_wistreambuf_);
    check_bool_trait<Trait, wlinkedbuf>(wlinkedbuf_);
#endif

    check_bool_trait<Trait, array<char>>(false);
    check_bool_trait<Trait, int>(false);
}

void bool_trait_test()
{
    // Test is_istream
    check_stream_trait<io::is_istream>(true, true, false, false, true, true,
        false, false, true, true, false, false, true, true, false, false, true,
        true, false, false, true, true, false, false, true, true, false, false,
        true, true, false, false, true, true, false, false, true, true, false,
        false, false, false);

    // Test is_ostream
    check_stream_trait<io::is_ostream>(false, false, true, true, true, true,
        false, false, false, false, true, true, true, true, false, false, false,
        false, true, true, true, true, false, false, false, false, true, true,
        true, true, false, false, false, false, true, true, true, true, false,
        false, false, false);

    // Test is_iostream
    check_stream_trait<io::is_iostream>(false, false, false, false, true, true,
        false, false, false, false, false, false, true, true, false, false,
        false, false, false, false, true, true, false, false, false, false,
        false, false, true, true, false, false, false, false, false, false,
        true, true, false, false, false, false);

    // Test is_streambuf
    check_stream_trait<io::is_streambuf>(false, false, false, false, false,
        false, true, true, false, false, false, false, false, false, true, true,
        false, false, false, false, false, false, true, true, false, false,
        false, false, false, false, true, true, false, false, false, false,
        false, false, true, true, true, true);

    // Test is_std_io
    check_stream_trait<io::is_std_io>(true, true, true, true, true, true, true,
        true, true, true, true, true, true, true, true, true, true, true, true,
        true, true, true, true, true, true, true, true, true, true, true, true,
        true, true, true, true, true, true, true, true, true, true, true);

    // Test is_std_file_device
    check_stream_trait<io::is_std_file_device>(false, false, false, false,
        false, false, false, false, true, true, true, true, true, true, true,
        true, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false);

    // Test is_std_string_device
    check_stream_trait<io::is_std_string_device>(false, false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, false, true, true, true, true, true, true, true, true, false,
        false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false);

    // Test is_iostreams_stream
    check_stream_trait<io::detail::is_iostreams_stream>(false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, true, true, true, true, true, true, false, false, false, false,
        false, false, false, false, false, false, false, false);

    // Test is_iostreams_stream_buffer
    check_stream_trait<io::detail::is_iostreams_stream_buffer>(false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, true, true,
        false, false, false, false, false, false, false, false, false, false);

    // Test is_filtering_stream
    check_stream_trait<io::detail::is_filtering_stream>(false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, true,
        true, true, true, true, true, false, false, false, false);

    // Test is_filtering_streambuf
    check_stream_trait<io::detail::is_filtering_streambuf>(false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, true, true, false, false);

    // Test is_iostreams
    check_stream_trait<io::detail::is_iostreams>(false, false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        true, true, true, true, true, true, true, true, true, true, true, true,
        true, true, true, true, false, false);
}

int main(int, char*[])
{
    bool_trait_test();
    return hpx::util::report_errors();
}
