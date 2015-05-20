//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_VARIANT_HPP
#define HPX_SERIALIZATION_VARIANT_HPP

#include <hpx/runtime/serialization/serialize.hpp>

#include <boost/variant.hpp>

namespace hpx { namespace serialization
{
    namespace detail
    {
        struct variant_save_visitor: boost::static_visitor<>
        {
            variant_save_visitor(output_archive& ar)
              : ar(ar)
            {}

            template <class T>
            void operator()(const T& value) const
            {
                ar << value;
            }

        private:
            output_archive& ar;
        };

        template <class... Args>
        struct load_helper_t{};

#if defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) || \
    defined(BOOST_VARIANT_DO_NOT_USE_VARIADIC_TEMPLATES) // intr. only in 1.56
        // overload for non-variadic version of variant library
        template <class Variant, class... Tail>
        BOOST_FORCEINLINE void load_helper(input_archive&, Variant&,
                int, load_helper_t<boost::detail::variant::void_, Tail...>)
        {
            HPX_ASSERT(false);
        }
#else
        // overload for variadic version of variant library
        template <class Variant>
        BOOST_FORCEINLINE void load_helper(input_archive&, Variant&,
                int, load_helper_t<>)
        {
            HPX_ASSERT(false);
        }
#endif
        template <class Variant, class Head, class... Tail>
        BOOST_FORCEINLINE void load_helper(input_archive& ar, Variant& var,
                int which, load_helper_t<Head, Tail...>)
        {
            if (which == 0)
            {
                Head value; // default ctor concept
                ar >> value;
                var = value;
            }
            else
            {
                load_helper(ar, var, which - 1, load_helper_t<Tail...>());
            }
        }
    }

    template <class... Args>
    void save(output_archive& ar, const boost::variant<Args...>& variant, unsigned)
    {
        int which = variant.which();
        ar << which;
        detail::variant_save_visitor visitor(ar);
        variant.apply_visitor(visitor);
    }

    template <class... Args>
    void load(input_archive& ar, boost::variant<Args...>& variant, unsigned)
    {
        int which;
        ar >> which;
        if (static_cast<std::size_t>(which) >= sizeof...(Args))
        {
            HPX_THROW_EXCEPTION(serialization_error
              , "load variant"
              , "which of loaded variant is greater then saved one");
        }

        detail::load_helper(ar, variant, which, detail::load_helper_t<Args...>());
    }

    HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE((template <class... Args>),
            (boost::variant<Args...>));
}}

#endif
