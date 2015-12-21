// convert_format.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details


#ifndef JT28092007_convert_format_HPP_DEFINED
#define JT28092007_convert_format_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/format/optimize.hpp>

namespace hpx { namespace util { namespace logging {

namespace tag {
    template<
            class string_ ,
            class param1 ,
            class param2 ,
            class param3 ,
            class param4 ,
            class param5 ,
            class param6 ,
            class param7 ,
            class param8 ,
            class param9 ,
            class param10> struct holder ;
}

namespace formatter {


/**
    @brief Allows format convertions
    - In case you're using a formatter that does not match your string type

    In case you want to use a formatter developed by someone else
    (for instance, a formatter provided by this lib),
    perhaps you're using another type of string to hold the message
    - thus, you need to provide a conversion function

    Example:
    FIXME

    --> convert_format::prepend

    explain that you can extend the following - since they're namespaces!!!
    so that you can "inject" your own write function
    in the convert_format::prepend/orwhatever namespace, and
    then it'll be automatically used!
*/
namespace convert {
    typedef hpx::util::logging::char_type char_type;
    typedef std::basic_string<char_type> string_type;
    typedef const char_type* char_array;

    template<class string> struct string_finder {
        typedef string type;
        typedef string original_type ;
        static const type& get(const original_type & str) { return str; }
    };
    template<class string> struct string_finder<
        ::hpx::util::logging::optimize::cache_string_one_str<string> > {
        typedef string type;
        typedef ::hpx::util::logging::optimize::cache_string_one_str<string>
            original_type ;
        static const type& get(const original_type & str) { return str; }
    };
    template<class string> struct string_finder<
        ::hpx::util::logging::optimize::cache_string_several_str<string> > {
        typedef string type;
        typedef ::hpx::util::logging::optimize::cache_string_several_str<string>
            original_type;
        static const type& get(const original_type & str) { return str; }
    };
    template<class string, class p1, class p2, class p3, class p4, class p5,
    class p6, class p7, class p8, class p9, class p10>
            struct string_finder< ::hpx::util::logging::tag::holder<string,
                p1,p2,p3,p4,p5,p6,p7,p8,p9,p10> > {
        typedef typename string_finder< string>::type type;
        typedef ::hpx::util::logging::tag::holder<string,p1,p2,p3,p4,p5,p6,
            p7,p8,p9,p10> original_type;

        // note: this needs 2 conversions - to string, and then to cache string
        static const type& get(const original_type & str) { return str; }
    };

    /**
    Example : write_time
    */
    namespace prepend {

        inline void write(char_array src, string_type & dest ) {
            const char_type * end = src;
            for ( ; *end; ++end) {}
            dest.insert( dest.begin(), src, end);
        }
        inline void write(const string_type & src, string_type & dest) {
            dest.insert( dest.begin(), src.begin(), src.end() );
        }
        template<class string> void write(const string_type & src,
            hpx::util::logging::optimize::cache_string_one_str<string> & dest) {
            dest.prepend_string(src);
        }
        template<class string> void write(const string_type & src,
            hpx::util::logging::optimize::cache_string_several_str<string> & dest) {
            dest.prepend_string(src);
        }

        template<class string_, class p1, class p2, class p3, class p4, class p5,
        class p6, class p7, class p8, class p9, class p10> void write(const string_type
            & src, ::hpx::util::logging::tag::holder<string_,p1,p2,p3,p4,p5,p6,p7,p8,
            p9,p10> & dest) {
            typedef typename use_default<string_, hold_string_type>::type string;
            write(src, static_cast<string&>(dest) );
        }



        template<class string> void write(char_array src,
            hpx::util::logging::optimize::cache_string_one_str<string> & dest) {
            dest.prepend_string(src);
        }
        template<class string> void write(char_array src,
            hpx::util::logging::optimize::cache_string_several_str<string> & dest) {
            dest.prepend_string(src);
        }

        template<class string_, class p1, class p2, class p3, class p4, class p5,
        class p6, class p7, class p8, class p9, class p10> void write(char_array src,
            ::hpx::util::logging::tag::holder<string_,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10>
            & dest) {
            typedef typename use_default<string_, hold_string_type>::type string;
            write(src, static_cast<string&>(dest) );
        }
    }

    /**
    */
    namespace append {
        inline void write(const string_type & src, string_type & dest) {
            dest += src;
        }
        template<class string> void write(const string_type & src,
            hpx::util::logging::optimize::cache_string_one_str<string> & dest) {
            dest.append_string(src);
        }
        template<class string> void write(const string_type & src,
            hpx::util::logging::optimize::cache_string_several_str<string> & dest) {
            dest.append_string(src);
        }
        template<class string_, class p1, class p2, class p3, class p4,
        class p5, class p6, class p7, class p8, class p9, class p10>
            void write(const string_type & src,
                ::hpx::util::logging::tag::holder<string_,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10>
                & dest) {
            typedef typename use_default<string_, hold_string_type>::type string;
            write(src, static_cast<string&>(dest) );
        }


        inline void write(char_array src, string_type & dest ) {
            dest += src;
        }
        template<class string> void write(char_array src,
            hpx::util::logging::optimize::cache_string_one_str<string> & dest) {
            dest.append_string(src);
        }
        template<class string> void write(char_array src,
            hpx::util::logging::optimize::cache_string_several_str<string> & dest) {
            dest.append_string(src);
        }
        template<class string_, class p1, class p2, class p3, class p4, class p5,
        class p6, class p7, class p8, class p9, class p10> void write(char_array src,
            ::hpx::util::logging::tag::holder<string_,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10>
            & dest) {
            typedef typename use_default<string_, hold_string_type>::type string;
            write(src, static_cast<string&>(dest) );
        }

    }

    /**
    */
    namespace modify {
        inline void write(const string_type & src, string_type & dest) {
            dest = src;
        }
        template<class string> void write(const string_type & src,
            hpx::util::logging::optimize::cache_string_one_str<string> & dest) {
            dest.set_string(src);
        }
        template<class string> void write(const string_type & src,
            hpx::util::logging::optimize::cache_string_several_str<string> & dest) {
            dest.set_string(src);
        }
        template<class string> void write(string_type & src,
            hpx::util::logging::optimize::cache_string_one_str<string> & dest) {
            dest.set_string_swap(src);
        }
        template<class string_, class p1, class p2, class p3, class p4, class p5,
        class p6, class p7, class p8, class p9, class p10> void write(const string_type
            & src, ::hpx::util::logging::tag::holder<string_,p1,p2,p3,p4,p5,p6,p7,
            p8,p9,p10> & dest) {
            typedef typename use_default<string_, hold_string_type>::type string;
            write(src, static_cast<string&>(dest) );
        }



        inline void write(char_array src, string_type & dest ) {
            dest = src;
        }
        template<class string> void write(char_array src,
            hpx::util::logging::optimize::cache_string_one_str<string> & dest) {
            dest.set_string(src);
        }
        template<class string> void write(char_array src,
            hpx::util::logging::optimize::cache_string_several_str<string> & dest) {
            dest.set_string(src);
        }
        template<class string_, class p1, class p2, class p3, class p4, class p5,
        class p6, class p7, class p8, class p9, class p10> void write(char_array src,
            ::hpx::util::logging::tag::holder<string_,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10>
            & dest) {
            typedef typename use_default<string_, hold_string_type>::type string;
            write(src, static_cast<string&>(dest) );
        }
    }
}


struct do_convert_format {
    typedef std::basic_string<char_type> string_type;

    struct prepend {
        template<class string>
                static inline const typename convert::string_finder<string>
                    ::type & get_underlying_string(const string & str) {
            return convert::string_finder<string>::get(str);
        }

        template<class string> static void write(const char_type * src, string & dest) {
            convert::prepend::write(src, dest);
        }

        template<class src_type, class string> static void write(const src_type & src,
            string & dest) {
            convert::prepend::write(src, dest);
        }
        template<class src_type, class string> static void write(src_type & src,
            string & dest) {
            convert::prepend::write(src, dest);
        }
    };

    struct append {
        template<class string>
                static inline const typename convert::string_finder<string>
                    ::type & get_underlying_string(const string & str) {
            return convert::string_finder<string>::get(str);
        }

        template<class string> static void write(const char_type * src, string & dest) {
            convert::append::write(src, dest);
        }

        template<class src_type, class string> static void write(const src_type & src,
            string & dest) {
            convert::append::write(src, dest);
        }
        template<class src_type, class string> static void write(src_type & src,
            string & dest) {
            convert::append::write(src, dest);
        }
    };

    struct modify {
        template<class string>
                static inline const typename convert::string_finder<string>
                    ::type & get_underlying_string(const string & str) {
            return convert::string_finder<string>::get(str);
        }

        template<class string> static void write(const char_type * src, string & dest) {
            convert::modify::write(src, dest);
        }

        template<class src_type, class string> static void write(const src_type & src,
            string & dest) {
            convert::modify::write(src, dest);
        }
        template<class src_type, class string> static void write(src_type & src,
            string & dest) {
            convert::modify::write(src, dest);
        }
    };
};

}}}}

#endif

