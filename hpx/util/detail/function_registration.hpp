//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_FUNCTION_REGISTRATION_HPP
#define HPX_UTIL_DETAIL_FUNCTION_REGISTRATION_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_intrusive_factory.hpp>
#include <hpx/traits/needs_automatic_registration.hpp>
#include <hpx/util/demangle_helper.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

#include <string>
#include <type_traits>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Function>
    struct get_function_name_impl
    {
        static char const* call()
#ifndef HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION
        ;
#else
        {
            // If you encounter this assert while compiling code, that means
            // that you have a HPX_UTIL_REGISTER_[UNIQUE_]FUNCTION macro
            // somewhere in a source file, but the header in which the function
            // is defined misses a HPX_UTIL_REGISTER_[UNIQUE_]FUNCTION_DECLARATION
            static_assert(
                traits::needs_automatic_registration<Function>::value,
                "HPX_UTIL_REGISTER_FUNCTION_DECLARATION missing");
            return util::type_id<Function>::typeid_.type_id();
        }
#endif
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Function>
    char const* get_function_name()
    {
        return get_function_name_impl<Function>::call();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable, typename T>
    struct function_registration
    {
        static void * create()
        {
            return const_cast<VTable*>(detail::get_vtable<VTable, T>());
        }

        function_registration()
        {
            typedef serializable_function_registration<VTable, T> vtable_pair;
            hpx::serialization::detail::polymorphic_intrusive_factory::instance().
                register_class(
                    detail::get_function_name<vtable_pair>()
                  , &function_registration::create
                );
        }
    };

    template <typename VTable>
    VTable const* get_vtable(std::string const& name)
    {
        return
            hpx::serialization::detail::polymorphic_intrusive_factory::instance().
                create<VTable const>(name);
    }
}}}

#endif
