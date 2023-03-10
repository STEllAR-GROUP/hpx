//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013-2023 Hartmut Kaiser
//  Copyright (c) 2014-2019 Agustin Berge
//  Copyright (c) 2017 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/functional/detail/basic_function.hpp>
#include <hpx/functional/detail/vtable/function_vtable.hpp>
#include <hpx/functional/detail/vtable/vtable.hpp>
#include <hpx/functional/serialization/detail/vtable/serializable_function_vtable.hpp>
#include <hpx/functional/serialization/detail/vtable/serializable_vtable.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <string>
#include <type_traits>
#include <utility>

namespace hpx::util::detail {

    template <bool Copyable, typename R, typename... Ts>
    class basic_function<R(Ts...), Copyable, /*Serializable*/ true>
      : public basic_function<R(Ts...), Copyable, /*Serializable*/ false>
    {
        using vtable = function_vtable<R(Ts...), Copyable>;
        using serializable_vtable = serializable_function_vtable<vtable>;
        using base_type = basic_function<R(Ts...), Copyable, false>;

    public:
        constexpr basic_function() noexcept
          : base_type()
          , serializable_vptr(nullptr)
        {
        }

        template <typename F>
        void assign(F&& f)
        {
            using target_type = std::decay_t<F>;

            base_type::assign(HPX_FORWARD(F, f));
            if (!base_type::empty())
            {
                serializable_vptr = get_serializable_vtable<target_type>();
            }
        }

        void swap(basic_function& f) noexcept
        {
            base_type::swap(f);
            std::swap(serializable_vptr, f.serializable_vptr);
        }

    private:
        friend class hpx::serialization::access;

        void save(
            serialization::output_archive& ar, unsigned const version) const
        {
            bool const is_empty = base_type::empty();
            ar << is_empty;
            if (!is_empty)
            {
                ar << std::string(serializable_vptr->name);
                serializable_vptr->save_object(object, ar, version);
            }
        }

        void load(serialization::input_archive& ar, unsigned const version)
        {
            base_type::reset();

            bool is_empty = false;
            ar >> is_empty;
            if (!is_empty)
            {
                std::string name;
                ar >> name;
                serializable_vptr =
                    detail::get_serializable_vtable<vtable>(HPX_MOVE(name));

                vptr = serializable_vptr->vptr;
                object = serializable_vptr->load_object(
                    storage, function_storage_size, ar, version);
            }
        }

        HPX_SERIALIZATION_SPLIT_MEMBER()

        template <typename T>
        static constexpr serializable_vtable const*
        get_serializable_vtable() noexcept
        {
            return detail::get_serializable_vtable<vtable, T>();
        }

    protected:
        using base_type::object;
        using base_type::storage;
        using base_type::vptr;
        serializable_vtable const* serializable_vptr;
    };
}    // namespace hpx::util::detail
