//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETECTED_HPP
#define HPX_UTIL_DETECTED_HPP

#include <hpx/config.hpp>
#include <hpx/util/always_void.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    // hpx::util::nonesuch is a class type used by hpx::util::detected_t to
    // indicate detection failure.
    struct nonesuch
    {
        nonesuch() = delete;
        ~nonesuch() = delete;
        nonesuch(nonesuch const&) = delete;
        void operator=(nonesuch const&) = delete;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Default, typename AlwaysVoid,
            template <typename...> class Op, typename ... Args>
        struct detector
        {
            using value_t = std::false_type;
            using type = Default;
        };

        template <typename Default, template <typename...> class Op,
            typename ... Args>
        struct detector<Default, typename always_void<Op<Args...> >::type,
            Op, Args...>
        {
            using value_t = std::true_type;
            using type = Op<Args...>;
        };
    }

    // The alias template is_detected is an alias for std::true_type if the
    // template-id Op<Args...> is valid; otherwise it is an alias for
    // std::false_type.
    template <template <typename...> class Op, typename ... Args>
    using is_detected =
        typename detail::detector<nonesuch, void, Op, Args...>::value_t;

    // The alias template detected_t is an alias for Op<Args...> if that
    // template-id is valid; otherwise it is an alias for the class
    // hpx::util::nonesuch.
    template <template <typename...> class Op, typename ... Args>
    using detected_t =
        typename detail::detector<nonesuch, void, Op, Args...>::type;

    // The alias template detected_or is an alias for an unspecified class type
    // with two public member typedefs value_t and type, which are defined as
    // follows:
    //
    // - If the template-id Op<Args...> is valid, then value_t is an alias for
    //   std::true_type, and type is an alias for Op<Args...>;
    // - Otherwise, value_t is an alias for std::false_type and type is an
    //   alias for Default.
    template <typename Default, template <typename...> class Op,
        typename ... Args>
    using detected_or =
        detail::detector<Default, void, Op, Args...>;

    template <typename Default, template <typename...> class Op, typename... Args>
    using detected_or_t = typename detected_or<Default, Op, Args...>::type;

    // The alias template is_detected_exact checks whether
    // detected_t<Op, Args...> is Expected.
    template <typename Expected, template <typename...> class Op,
        typename... Args>
    using is_detected_exact =
        std::is_same<Expected, detected_t<Op, Args...> >;

    // The alias template is_detected_convertible checks whether
    // detected_t<Op, Args...> is convertible to To.
    template <typename To, template <typename...> class Op, typename... Args>
    using is_detected_convertible =
        std::is_convertible<detected_t<Op, Args...>, To>;
}}

#endif
