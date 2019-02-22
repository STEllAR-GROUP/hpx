//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2019 Agustin Berge
//  Copyright (c) 2017 Google
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/util/detail/basic_function.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/traits/get_function_address.hpp>
#include <hpx/traits/get_function_annotation.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/detail/empty_function.hpp>
#include <hpx/util/detail/vtable/function_vtable.hpp>
#include <hpx/util/detail/vtable/serializable_function_vtable.hpp>
#include <hpx/util/detail/vtable/serializable_vtable.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>
#include <hpx/util/itt_notify.hpp>

#include <cstddef>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    function_base::function_base(
        function_base const& other,
        vtable const* empty_vtable)
      : vptr(other.vptr)
      , object(other.object)
    {
        if (other.object != nullptr)
        {
            object = vptr->copy(
                storage, detail::function_storage_size,
                other.object, /*destroy*/false);
        }
    }

    function_base::function_base(
        function_base&& other,
        vtable const* empty_vptr) noexcept
      : vptr(other.vptr)
      , object(other.object)
    {
        if (object == &other.storage)
        {
            std::memcpy(storage, other.storage, function_storage_size);
            object = &storage;
        }
        other.vptr = empty_vptr;
        other.object = nullptr;
    }

    function_base::~function_base()
    {
        destroy();
    }

    void function_base::op_assign(
        function_base const& other,
        vtable const* empty_vtable)
    {
        if (vptr == other.vptr)
        {
            if (this != &other && object)
            {
                HPX_ASSERT(other.object != nullptr);
                // reuse object storage
                object = vptr->copy(
                    object, -1,
                    other.object, /*destroy*/true);
            }
        } else {
            destroy();
            vptr = other.vptr;
            if (other.object != nullptr)
            {
                object = vptr->copy(
                    storage, detail::function_storage_size,
                    other.object, /*destroy*/false);
            } else {
                object = nullptr;
            }
        }
    }

    void function_base::op_assign(
        function_base&& other,
        vtable const* empty_vtable) noexcept
    {
        if (this != &other)
        {
            swap(other);
            other.reset(empty_vtable);
        }
    }

    void function_base::destroy() noexcept
    {
        if (object != nullptr)
        {
            vptr->deallocate(
                object, function_storage_size,
                /*destroy*/true);
        }
    }

    void function_base::reset(vtable const* empty_vptr) noexcept
    {
        destroy();
        vptr = empty_vptr;
        object = nullptr;
    }

    void function_base::swap(function_base& f) noexcept
    {
        std::swap(vptr, f.vptr);
        std::swap(object, f.object);
        std::swap(storage, f.storage);
        if (object == &f.storage)
            object = &storage;
        if (f.object == &storage)
            f.object = &f.storage;
    }

    std::size_t function_base::get_function_address() const
    {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
        return vptr->get_function_address(object);
#else
        return 0;
#endif
    }

    char const* function_base::get_function_annotation() const
    {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
        return vptr->get_function_annotation(object);
#else
        return nullptr;
#endif
    }

    util::itt::string_handle function_base::get_function_annotation_itt() const
    {
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
        return vptr->get_function_annotation_itt(object);
#else
        return util::itt::string_handle{};
#endif
    }
}}}
