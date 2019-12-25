// Copyright Kevlin Henney, 2000-2005.
// Copyright Alexander Nasonov, 2006-2010.
// Copyright Antony Polukhin, 2011-2019.
// Copyright Agustin Berge, 2019.
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// what:  lexical_cast custom keyword cast
// who:   contributed by Kevlin Henney,
//        enhanced with contributions from Terje Slettebo,
//        with additional fixes and suggestions from Gennaro Prota,
//        Beman Dawes, Dave Abrahams, Daryle Walker, Peter Dimov,
//        Alexander Nasonov, Antony Polukhin, Justin Viiret, Michael Hofmann,
//        Cheng Yang, Matthew Bradbury, David W. Birdsall, Pavel Korzh and
//        other Boosters
// when:  November 2000, March 2003, June 2005, June 2006, March 2011 - 2014,
// Nowember 2016

#ifndef HPX_LEXICAL_CAST_DETAIL_CONVERTER_LEXICAL_STREAMS_HPP
#define HPX_LEXICAL_CAST_DETAIL_CONVERTER_LEXICAL_STREAMS_HPP

#include <hpx/config.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <istream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

#include <hpx/lexical_cast/detail/basic_pointerbuf.hpp>
#include <hpx/lexical_cast/detail/cstring_wrapper.hpp>
#include <hpx/lexical_cast/detail/inf_nan.hpp>
#include <hpx/lexical_cast/detail/lcast_char_constants.hpp>
#include <hpx/lexical_cast/detail/lcast_precision.hpp>
#include <hpx/lexical_cast/detail/lcast_unsigned_converters.hpp>

namespace hpx { namespace util {

    namespace detail    // basic_unlockedbuf
    {
        // acts as a stream buffer which wraps around a pair of pointers
        // and gives acces to internals
        template <class BufferType, class CharT>
        class basic_unlockedbuf : public basic_pointerbuf<CharT, BufferType>
        {
        public:
            typedef basic_pointerbuf<CharT, BufferType> base_type;
            typedef typename base_type::streamsize streamsize;

            using base_type::pbase;
            using base_type::pptr;
            using base_type::setbuf;
        };
    }    // namespace detail

    namespace detail {
        struct do_not_construct_out_buffer_t
        {
        };
        struct do_not_construct_out_stream_t
        {
            do_not_construct_out_stream_t(do_not_construct_out_buffer_t*) {}
        };

        template <class CharT, class Traits>
        struct out_stream_helper_trait
        {
            typedef std::ostream out_stream_t;
            typedef basic_unlockedbuf<std::stringbuf, char> stringbuffer_t;
            typedef basic_unlockedbuf<std::streambuf, char> buffer_t;
        };
    }    // namespace detail

    namespace detail    // optimized stream wrappers
    {
        template <class CharT, class Traits, bool RequiresStringbuffer,
            std::size_t CharacterBufferSize>
        class lexical_istream_limited_src
        {
            HPX_NON_COPYABLE(lexical_istream_limited_src);

            typedef typename std::conditional<RequiresStringbuffer,
                typename out_stream_helper_trait<CharT, Traits>::out_stream_t,
                do_not_construct_out_stream_t>::type deduced_out_stream_t;

            typedef typename std::conditional<RequiresStringbuffer,
                typename out_stream_helper_trait<CharT, Traits>::stringbuffer_t,
                do_not_construct_out_buffer_t>::type deduced_out_buffer_t;

            deduced_out_buffer_t out_buffer;
            deduced_out_stream_t out_stream;
            CharT buffer[CharacterBufferSize];

            // After the `operator <<`  finishes, `[start, finish)` is
            // the range to output by `operator >>`
            const CharT* start;
            const CharT* finish;

        public:
            lexical_istream_limited_src() noexcept
              : out_buffer()
              , out_stream(&out_buffer)
              , start(buffer)
              , finish(buffer + CharacterBufferSize)
            {
            }

            const CharT* cbegin() const noexcept
            {
                return start;
            }

            const CharT* cend() const noexcept
            {
                return finish;
            }

        private:
            /* HELPER FUNCTIONS FOR OPERATORS << ( ... ) */
            bool shl_char(CharT ch) noexcept
            {
                Traits::assign(buffer[0], ch);
                finish = start + 1;
                return true;
            }

            bool shl_char_array(CharT const* str_value) noexcept
            {
                start = str_value;
                finish = start + Traits::length(str_value);
                return true;
            }

            template <class T>
            bool shl_char_array(T const* str_value)
            {
                static_assert(sizeof(T) <= sizeof(CharT),
                    "hpx::util::lexical_cast does not support narrowing of "
                    "character types."
                    "Use boost::locale instead");
                return shl_input_streamable(str_value);
            }

            bool shl_char_array_limited(
                CharT const* str, std::size_t max_size) noexcept
            {
                start = str;
                finish =
                    std::find(start, start + max_size, Traits::to_char_type(0));
                return true;
            }

            template <typename InputStreamable>
            bool shl_input_streamable(InputStreamable& input)
            {
                static_assert(std::is_same<char, CharT>::value,
                    "hpx::util::lexical_cast can not convert.");

                out_stream.exceptions(std::ios::badbit);
                try
                {
                    bool const result = !(out_stream << input).fail();
                    const deduced_out_buffer_t* const p =
                        static_cast<deduced_out_buffer_t*>(out_stream.rdbuf());
                    start = p->pbase();
                    finish = p->pptr();
                    return result;
                }
                catch (const ::std::ios_base::failure& /*f*/)
                {
                    return false;
                }
            }

            template <class T>
            inline bool shl_unsigned(const T n)
            {
                CharT* tmp_finish = buffer + CharacterBufferSize;
                start = lcast_put_unsigned<Traits, T, CharT>(n, tmp_finish)
                            .convert();
                finish = tmp_finish;
                return true;
            }

            template <class T>
            inline bool shl_signed(const T n)
            {
                CharT* tmp_finish = buffer + CharacterBufferSize;
                typedef typename std::make_unsigned<T>::type utype;
                CharT* tmp_start = lcast_put_unsigned<Traits, utype, CharT>(
                    lcast_to_unsigned(n), tmp_finish)
                                       .convert();
                if (n < 0)
                {
                    --tmp_start;
                    CharT const minus = lcast_char_constants<CharT>::minus;
                    Traits::assign(*tmp_start, minus);
                }
                start = tmp_start;
                finish = tmp_finish;
                return true;
            }

            template <class T, class SomeCharT>
            bool shl_real_type(const T& val, SomeCharT* /*begin*/)
            {
                lcast_set_precision(out_stream, &val);
                return shl_input_streamable(val);
            }

            bool shl_real_type(float val, char* begin)
            {
                using namespace std;
                const double val_as_double = val;
                finish = start +
#if defined(_MSC_VER) && (_MSC_VER >= 1400) && !defined(__SGI_STL_PORT) &&     \
    !defined(_STLPORT_VERSION)
                    sprintf_s(begin, CharacterBufferSize,
#else
                    sprintf(begin,
#endif
                        "%.*g",
                        static_cast<int>(detail::lcast_get_precision<float>()),
                        val_as_double);
                return finish > start;
            }

            bool shl_real_type(double val, char* begin)
            {
                using namespace std;
                finish = start +
#if defined(_MSC_VER) && (_MSC_VER >= 1400) && !defined(__SGI_STL_PORT) &&     \
    !defined(_STLPORT_VERSION)
                    sprintf_s(begin, CharacterBufferSize,
#else
                    sprintf(begin,
#endif
                        "%.*g",
                        static_cast<int>(detail::lcast_get_precision<double>()),
                        val);
                return finish > start;
            }

#ifndef __MINGW32__
            bool shl_real_type(long double val, char* begin)
            {
                using namespace std;
                finish = start +
#if defined(_MSC_VER) && (_MSC_VER >= 1400) && !defined(__SGI_STL_PORT) &&     \
    !defined(_STLPORT_VERSION)
                    sprintf_s(begin, CharacterBufferSize,
#else
                    sprintf(begin,
#endif
                        "%.*Lg",
                        static_cast<int>(
                            detail::lcast_get_precision<long double>()),
                        val);
                return finish > start;
            }
#endif

            template <class T>
            bool shl_real(T val)
            {
                CharT* tmp_finish = buffer + CharacterBufferSize;
                if (put_inf_nan(buffer, tmp_finish, val))
                {
                    finish = tmp_finish;
                    return true;
                }

                return shl_real_type(val, static_cast<CharT*>(buffer));
            }

            /* OPERATORS << ( ... ) */
        public:
            template <class Alloc>
            bool operator<<(
                std::basic_string<CharT, Traits, Alloc> const& str) noexcept
            {
                start = str.data();
                finish = start + str.length();
                return true;
            }

            bool operator<<(bool value) noexcept
            {
                CharT const czero = lcast_char_constants<CharT>::zero;
                Traits::assign(buffer[0], Traits::to_char_type(czero + value));
                finish = start + 1;
                return true;
            }

            bool operator<<(const cstring_wrapper<CharT>& rng) noexcept
            {
                start = rng.data;
                finish = start + rng.length;
                return true;
            }

            bool operator<<(char ch)
            {
                return shl_char(ch);
            }
            bool operator<<(unsigned char ch)
            {
                return ((*this) << static_cast<char>(ch));
            }
            bool operator<<(signed char ch)
            {
                return ((*this) << static_cast<char>(ch));
            }
            bool operator<<(unsigned char const* ch)
            {
                return ((*this) << reinterpret_cast<char const*>(ch));
            }
            bool operator<<(unsigned char* ch)
            {
                return ((*this) << reinterpret_cast<char*>(ch));
            }
            bool operator<<(signed char const* ch)
            {
                return ((*this) << reinterpret_cast<char const*>(ch));
            }
            bool operator<<(signed char* ch)
            {
                return ((*this) << reinterpret_cast<char*>(ch));
            }
            bool operator<<(char const* str_value)
            {
                return shl_char_array(str_value);
            }
            bool operator<<(char* str_value)
            {
                return shl_char_array(str_value);
            }
            bool operator<<(short n)
            {
                return shl_signed(n);
            }
            bool operator<<(int n)
            {
                return shl_signed(n);
            }
            bool operator<<(long n)
            {
                return shl_signed(n);
            }
            bool operator<<(long long n)
            {
                return shl_signed(n);
            }
            bool operator<<(unsigned short n)
            {
                return shl_unsigned(n);
            }
            bool operator<<(unsigned int n)
            {
                return shl_unsigned(n);
            }
            bool operator<<(unsigned long n)
            {
                return shl_unsigned(n);
            }
            bool operator<<(unsigned long long n)
            {
                return shl_unsigned(n);
            }
            bool operator<<(float val)
            {
                return shl_real(val);
            }
            bool operator<<(double val)
            {
                return shl_real(val);
            }
            bool operator<<(long double val)
            {
#ifndef __MINGW32__
                return shl_real(val);
#else
                return shl_real(static_cast<double>(val));
#endif
            }
            template <class C, std::size_t N>
            bool operator<<(std::array<C, N> const& input) noexcept
            {
                return shl_char_array_limited(input.data(), N);
            }
            template <class InStreamable>
            bool operator<<(const InStreamable& input)
            {
                return shl_input_streamable(input);
            }
        };

        template <class CharT, class Traits>
        class lexical_ostream_limited_src
        {
            HPX_NON_COPYABLE(lexical_ostream_limited_src);

            //`[start, finish)` is the range to output by `operator >>`
            const CharT* start;
            const CharT* const finish;

        public:
            lexical_ostream_limited_src(
                const CharT* begin, const CharT* end) noexcept
              : start(begin)
              , finish(end)
            {
            }

            /* HELPER FUNCTIONS FOR OPERATORS >> ( ... ) */
        private:
            template <typename Type>
            bool shr_unsigned(Type& output)
            {
                if (start == finish)
                    return false;
                CharT const minus = lcast_char_constants<CharT>::minus;
                CharT const plus = lcast_char_constants<CharT>::plus;
                bool const has_minus = Traits::eq(minus, *start);

                // We won`t use `start' any more, so no need in decrementing it after
                if (has_minus || Traits::eq(plus, *start))
                {
                    ++start;
                }

                bool const succeed = lcast_ret_unsigned<Traits, Type, CharT>(
                    output, start, finish)
                                         .convert();

                if (has_minus)
                {
                    output = static_cast<Type>(0u - output);
                }

                return succeed;
            }

            template <typename Type>
            bool shr_signed(Type& output)
            {
                if (start == finish)
                    return false;
                CharT const minus = lcast_char_constants<CharT>::minus;
                CharT const plus = lcast_char_constants<CharT>::plus;
                typedef typename std::make_unsigned<Type>::type utype;
                utype out_tmp = 0;
                bool const has_minus = Traits::eq(minus, *start);

                // We won`t use `start' any more, so no need in decrementing it after
                if (has_minus || Traits::eq(plus, *start))
                {
                    ++start;
                }

                bool succeed = lcast_ret_unsigned<Traits, utype, CharT>(
                    out_tmp, start, finish)
                                   .convert();
                if (has_minus)
                {
                    utype const comp_val = (static_cast<utype>(1)
                        << std::numeric_limits<Type>::digits);
                    succeed = succeed && out_tmp <= comp_val;
                    output = static_cast<Type>(0u - out_tmp);
                }
                else
                {
                    utype const comp_val =
                        static_cast<utype>((std::numeric_limits<Type>::max)());
                    succeed = succeed && out_tmp <= comp_val;
                    output = static_cast<Type>(out_tmp);
                }
                return succeed;
            }

            template <typename InputStreamable>
            bool shr_using_base_class(InputStreamable& output)
            {
                static_assert(!std::is_pointer<InputStreamable>::value,
                    "hpx::util::lexical_cast can not convert to pointers");

                static_assert(std::is_same<char, CharT>::value,
                    "hpx::util::lexical_cast can not convert.");

                typedef
                    typename out_stream_helper_trait<CharT, Traits>::buffer_t
                        buffer_t;
                buffer_t buf;
                // Usually `istream` and `basic_istream` do not modify
                // content of buffer; `buffer_t` assures that this is true
                buf.setbuf(const_cast<CharT*>(start),
                    static_cast<typename buffer_t::streamsize>(finish - start));
                std::istream stream(&buf);

                stream.exceptions(std::ios::badbit);
                try
                {
                    stream.unsetf(std::ios::skipws);
                    lcast_set_precision(
                        stream, static_cast<InputStreamable*>(nullptr));

                    return (stream >> output) &&
                        (stream.get() == Traits::eof());
                }
                catch (const ::std::ios_base::failure& /*f*/)
                {
                    return false;
                }
            }

            template <class T>
            inline bool shr_xchar(T& output) noexcept
            {
                static_assert(sizeof(CharT) == sizeof(T),
                    "hpx::util::lexical_cast does not support narrowing of "
                    "character types."
                    "Use boost::locale instead");
                bool const ok = (finish - start == 1);
                if (ok)
                {
                    CharT out;
                    Traits::assign(out, *start);
                    output = static_cast<T>(out);
                }
                return ok;
            }

            template <std::size_t N, class ArrayT>
            bool shr_std_array(ArrayT& output) noexcept
            {
                using namespace std;
                const std::size_t size =
                    static_cast<std::size_t>(finish - start);
                if (size > N - 1)
                {    // `-1` because we need to store \0 at the end
                    return false;
                }

                memcpy(&output[0], start, size * sizeof(CharT));
                output[size] = Traits::to_char_type(0);
                return true;
            }

            /* OPERATORS >> ( ... ) */
        public:
            bool operator>>(unsigned short& output)
            {
                return shr_unsigned(output);
            }
            bool operator>>(unsigned int& output)
            {
                return shr_unsigned(output);
            }
            bool operator>>(unsigned long int& output)
            {
                return shr_unsigned(output);
            }
            bool operator>>(unsigned long long& output)
            {
                return shr_unsigned(output);
            }
            bool operator>>(short& output)
            {
                return shr_signed(output);
            }
            bool operator>>(int& output)
            {
                return shr_signed(output);
            }
            bool operator>>(long int& output)
            {
                return shr_signed(output);
            }
            bool operator>>(long long& output)
            {
                return shr_signed(output);
            }
            bool operator>>(char& output)
            {
                return shr_xchar(output);
            }
            bool operator>>(unsigned char& output)
            {
                return shr_xchar(output);
            }
            bool operator>>(signed char& output)
            {
                return shr_xchar(output);
            }
            template <class Alloc>
            bool operator>>(std::basic_string<CharT, Traits, Alloc>& str)
            {
                str.assign(start, finish);
                return true;
            }
            template <class C, std::size_t N>
            bool operator>>(std::array<C, N>& output) noexcept
            {
                return shr_std_array<N>(output);
            }
            bool operator>>(bool& output) noexcept
            {
                output =
                    false;    // Suppress warning about uninitalized variable

                if (start == finish)
                    return false;
                CharT const zero = lcast_char_constants<CharT>::zero;
                CharT const plus = lcast_char_constants<CharT>::plus;
                CharT const minus = lcast_char_constants<CharT>::minus;

                const CharT* const dec_finish = finish - 1;
                output = Traits::eq(*dec_finish, zero + 1);
                if (!output && !Traits::eq(*dec_finish, zero))
                {
                    return false;    // Does not ends on '0' or '1'
                }

                if (start == dec_finish)
                    return true;

                // We may have sign at the beginning
                if (Traits::eq(plus, *start) ||
                    (Traits::eq(minus, *start) && !output))
                {
                    ++start;
                }

                // Skipping zeros
                while (start != dec_finish)
                {
                    if (!Traits::eq(zero, *start))
                    {
                        return false;    // Not a zero => error
                    }

                    ++start;
                }

                return true;
            }

        private:
            // Not optimised converter
            template <class T>
            bool float_types_converter_internal(T& output)
            {
                if (parse_inf_nan(start, finish, output))
                    return true;
                bool const return_value = shr_using_base_class(output);

                /* Some compilers and libraries successfully
                 * parse 'inf', 'INFINITY', '1.0E', '1.0E-'...
                 * We are trying to provide a unified behaviour,
                 * so we just forbid such conversions (as some
                 * of the most popular compilers/libraries do)
                 * */
                CharT const minus = lcast_char_constants<CharT>::minus;
                CharT const plus = lcast_char_constants<CharT>::plus;
                CharT const capital_e = lcast_char_constants<CharT>::capital_e;
                CharT const lowercase_e =
                    lcast_char_constants<CharT>::lowercase_e;
                if (return_value &&
                    (Traits::eq(*(finish - 1), lowercase_e)        // 1.0e
                        || Traits::eq(*(finish - 1), capital_e)    // 1.0E
                        || Traits::eq(*(finish - 1), minus)    // 1.0e- or 1.0E-
                        || Traits::eq(*(finish - 1), plus)     // 1.0e+ or 1.0E+
                        ))
                    return false;

                return return_value;
            }

        public:
            bool operator>>(float& output)
            {
                return float_types_converter_internal(output);
            }
            bool operator>>(double& output)
            {
                return float_types_converter_internal(output);
            }
            bool operator>>(long double& output)
            {
                return float_types_converter_internal(output);
            }

            // Generic istream-based algorithm.
            // lcast_streambuf_for_target<InputStreamable>::value is true.
            template <typename InputStreamable>
            bool operator>>(InputStreamable& output)
            {
                return shr_using_base_class(output);
            }
        };
    }    // namespace detail
}}       // namespace hpx::util

#endif    // HPX_LEXICAL_CAST_DETAIL_CONVERTER_LEXICAL_HPP
