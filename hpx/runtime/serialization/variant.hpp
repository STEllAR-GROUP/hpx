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
            void operator()(const T& value)
            {
                ar << value;
            }

        private:
            output_archive& ar;
        };

        template <class... Args>
        struct load_helper_t{};

        template <class Variant, class Head, class... Args>
        BOOST_FORCEINLINE void load_helper(input_archive& ar, Variant& var,
                load_helper_t<Head, boost::detail::variant::void_, Args...>)
        {
            Head value; //default ctor concept
            ar >> value;
            var = value;
        }

        template <class Variant, class Head1, class Head2, class... Tail>
        BOOST_FORCEINLINE void load_helper(input_archive& ar, Variant& var,
                load_helper_t<Head1, Head2, Tail...>)
        {
            load_helper(ar, var, load_helper_t<Head2, Tail...>());
        }
    }

    template <class... Args>
    void serialize(output_archive& ar, boost::variant<Args...>& variant, unsigned)
    {
        detail::variant_save_visitor visitor(ar);
        variant.apply_visitor(visitor);
    }

    template <class... Args>
    void serialize(input_archive& ar, boost::variant<Args...>& variant, unsigned)
    {
        detail::load_helper(ar, variant, detail::load_helper_t<Args...>());
    }
}}

#endif
