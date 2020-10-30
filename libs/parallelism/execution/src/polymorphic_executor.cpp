//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution/executors/polymorphic_executor.hpp>
#include <hpx/modules/errors.hpp>

#include <cstddef>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace execution { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    HPX_NORETURN void throw_bad_polymorphic_executor()
    {
        hpx::throw_exception(bad_function_call,
            "empty polymorphic_executor object should not be used",
            "polymorphic_executor::operator()");
    }

    ///////////////////////////////////////////////////////////////////////////
    polymorphic_executor_base::polymorphic_executor_base(
        polymorphic_executor_base const& other,
        vtable const* /* empty_vtable */)
      : vptr(other.vptr)
      , object(other.object)
    {
        if (other.object != nullptr)
        {
            object = vptr->copy(storage, polymorphic_executor_storage_size,
                other.object, false);
        }
    }

    polymorphic_executor_base::polymorphic_executor_base(
        polymorphic_executor_base&& other, vtable const* empty_vptr) noexcept
      : vptr(other.vptr)
      , object(other.object)
    {
        if (object == &other.storage)
        {
            std::memcpy(
                storage, other.storage, polymorphic_executor_storage_size);
            object = &storage;
        }
        other.vptr = empty_vptr;
        other.object = nullptr;
    }

    polymorphic_executor_base::~polymorphic_executor_base()
    {
        destroy();
    }

    void polymorphic_executor_base::op_assign(
        polymorphic_executor_base const& other,
        vtable const* /* empty_vtable */)
    {
        if (vptr == other.vptr)
        {
            if (this != &other && object)
            {
                // reuse object storage
                HPX_ASSERT(other.object != nullptr);
                object = vptr->copy(object, -1, other.object, true);
            }
        }
        else
        {
            destroy();
            vptr = other.vptr;
            if (other.object != nullptr)
            {
                object = vptr->copy(storage, polymorphic_executor_storage_size,
                    other.object, false);
            }
            else
            {
                object = nullptr;
            }
        }
    }

    void polymorphic_executor_base::op_assign(
        polymorphic_executor_base&& other, vtable const* empty_vtable) noexcept
    {
        if (this != &other)
        {
            swap(other);
            other.reset(empty_vtable);
        }
    }

    void polymorphic_executor_base::destroy() noexcept
    {
        if (object != nullptr)
        {
            vptr->deallocate(object, polymorphic_executor_storage_size, true);
        }
    }

    void polymorphic_executor_base::reset(vtable const* empty_vptr) noexcept
    {
        destroy();
        vptr = empty_vptr;
        object = nullptr;
    }

    void polymorphic_executor_base::swap(polymorphic_executor_base& f) noexcept
    {
        std::swap(vptr, f.vptr);
        std::swap(object, f.object);
        std::swap(storage, f.storage);
        if (object == &f.storage)
            object = &storage;
        if (f.object == &storage)
            f.object = &f.storage;
    }
}}}}    // namespace hpx::parallel::execution::detail
