// logger_to_writer.hpp

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


#ifndef JT28092007_logger_to_writer_HPP_DEFINED
#define JT28092007_logger_to_writer_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/format_fwd.hpp>
#include <memory>

namespace hpx { namespace util { namespace logging {

// ... the hard part - for logger_format_write

namespace formatter {
    template< class arg_type, class ptr_type_> struct base ;
}

namespace destination {
    template< class arg_type, class ptr_type_> struct base ;
}

namespace msg_route {
    template<class formatter_base, class destination_base,
    class lock_resource > struct simple ;
    template<class formatter_base, class destination_base,
    class lock_resource, class formatter_array,
    class destination_array> class with_route;
}

namespace format_and_write {
    template<class msg_type> struct simple ;

    template<class formatter_base, class destination_base,
    class msg_type> struct use_cache ;
}
namespace array {
    template <class base_type, class mutex > class shared_ptr_holder ;
}

namespace writer {
    template<
            class formatter_base,
            class destination_base,
            class lock_resource ,
            class apply_format_and_write ,
            class router_type ,
            class formatter_array ,
            class destination_array >
    struct format_write ;
}


namespace detail {

    //////////// find_format_write_params
    template<class string, class formatter_base, class destination_base,
        class lock_resource> struct find_format_write_params {
        typedef typename ::hpx::util::logging::format_and_write::simple<string>
            apply_format_and_write ;
        typedef typename ::hpx::util::logging::msg_route::simple<formatter_base,
            destination_base, lock_resource> router_type;
    };

    template<class string_type, class formatter_base, class destination_base,
        class lock_resource>
        struct find_format_write_params< typename hpx::util
            ::logging::optimize::cache_string_several_str<string_type>,
            formatter_base, destination_base, lock_resource>
    {
        typedef typename hpx::util::logging::optimize
            ::cache_string_several_str<string_type> cache_string;
        typedef typename hpx::util::logging::format_and_write::use_cache<formatter_base,
            destination_base, cache_string> apply_format_and_write ;

        typedef hpx::util::logging::array::shared_ptr_holder<formatter_base,
            hpx::util::logging::threading::mutex > formatter_array ;
        typedef hpx::util::logging::array::shared_ptr_holder<destination_base,
            hpx::util::logging::threading::mutex > destination_array ;

        typedef typename msg_route::with_route<formatter_base,
            destination_base, lock_resource, formatter_array,
            destination_array > router_type;
    };



    ///////////// find_writer_with_thread_safety
    template<class thread_safety,
    class format_write> struct find_writer_with_thread_safety {
        // for default_
#ifndef HPX_HAVE_LOG_NO_TS
        // use ts_write
        typedef writer::ts_write<format_write> type;
#else
        typedef format_write type;
#endif
    };

    template<class format_write> struct find_writer_with_thread_safety<hpx
        ::util::logging::writer::threading::no_ts,format_write> {
        typedef format_write type;
    };

    template<class format_write> struct find_writer_with_thread_safety<hpx
            ::util::logging::writer::threading::ts_write,format_write> {
        typedef writer::ts_write<format_write> type;
    };

    template<class format_write> struct find_writer_with_thread_safety<hpx
            ::util::logging::writer::threading::on_dedicated_thread,format_write> {
        typedef typename detail::to_override<format_write>::type override_;
        typedef typename formatter::msg_type<override_>::type msg_type;

        typedef writer::on_dedicated_thread<msg_type, format_write> type;
    };
}


namespace detail {
    template< class format_base_type , class destination_base_type ,
    class lock_resource, class thread_safety > struct format_find_writer {

        typedef typename detail::to_override<format_base_type>::type override_;
        typedef typename ::hpx::util::logging::formatter::msg_type<override_>
            ::type format_msg_type;
        typedef typename ::hpx::util::logging::destination::msg_type<override_>
            ::type destination_msg_type;
        typedef typename ::hpx::util::logging::types<override_>
            ::lock_resource default_lock_resource;

        typedef ::hpx::util::logging::formatter::base< default_, default_ >
            default_format_base;
        typedef ::hpx::util::logging::destination::base< default_, default_ >
            default_destination_base;

        typedef typename use_default<format_base_type, default_format_base >
            ::type format_base;
        typedef typename use_default<destination_base_type, default_destination_base >
            ::type destination_base;
        typedef typename use_default<lock_resource, default_lock_resource >
            ::type lock_resource_type;



        typedef typename detail::find_format_write_params<format_msg_type,
            format_base, destination_base, lock_resource_type >
            ::apply_format_and_write apply_format_and_write;
        typedef typename detail::find_format_write_params<format_msg_type,
            format_base, destination_base, lock_resource_type >::router_type router_type;

        typedef ::hpx::util::logging::array::shared_ptr_holder<format_base,
            hpx::util::logging::threading::mutex > formatter_array ;
        typedef ::hpx::util::logging::array::shared_ptr_holder<destination_base,
            hpx::util::logging::threading::mutex > destination_array ;

        // now find the writer based on thread safety
        typedef writer::format_write<
            format_base,
            destination_base,
            lock_resource_type,
            apply_format_and_write,
            router_type,
            formatter_array,
            destination_array > format_write_type;
        typedef typename find_writer_with_thread_safety<thread_safety,
            format_write_type>::type type;
    };
}

}}}

#endif

