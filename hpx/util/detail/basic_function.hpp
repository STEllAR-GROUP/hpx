//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2019 Agustin Berge
//  Copyright (c) 2017 Google
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_BASIC_FUNCTION_HPP
#define HPX_UTIL_DETAIL_BASIC_FUNCTION_HPP

#include <hpx/config.hpp>
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

#include <cstddef>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace util { namespace detail
{
    static const std::size_t function_storage_size = 3 * sizeof(void*);

    ///////////////////////////////////////////////////////////////////////////
    class function_base
    {
        using vtable = function_base_vtable;

    public:
        HPX_CONSTEXPR function_base(
            function_base_vtable const* empty_vptr) noexcept
          : vptr(empty_vptr)
          , object(nullptr)
          , storage_init()
        {}

        function_base(
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

        function_base(
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

        ~function_base()
        {
            destroy();
        }

        void op_assign(
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

        void op_assign(
            function_base&& other,
            vtable const* empty_vtable) noexcept
        {
            if (this != &other)
            {
                swap(other);
                other.reset(empty_vtable);
            }
        }

        void destroy() noexcept
        {
            if (object != nullptr)
            {
                vptr->deallocate(
                    object, function_storage_size,
                    /*destroy*/true);
            }
        }

        void reset(vtable const* empty_vptr) noexcept
        {
            destroy();
            vptr = empty_vptr;
            object = nullptr;
        }

        void swap(function_base& f) noexcept
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            std::swap(storage, f.storage);
            if (object == &f.storage)
                object = &storage;
            if (f.object == &storage)
                f.object = &f.storage;
        }

        bool empty() const noexcept
        {
            return object == nullptr;
        }

        explicit operator bool() const noexcept
        {
            return !empty();
        }

        std::size_t get_function_address() const
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            return vptr->get_function_address(object);
#else
            return 0;
#endif
        }

        char const* get_function_annotation() const
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            return vptr->get_function_annotation(object);
#else
            return nullptr;
#endif
        }

        util::itt::string_handle get_function_annotation_itt() const
        {
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
            return vptr->get_function_annotation_itt(object);
#else
            return util::itt::string_handle{};
#endif
        }

    protected:
        vtable const* vptr;
        void* object;
        union {
            char storage_init;
            mutable unsigned char storage[function_storage_size];
        };
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    HPX_CONSTEXPR bool is_empty_function(F* fp) noexcept
    {
        return fp == nullptr;
    }

    template <typename T, typename C>
    HPX_CONSTEXPR bool is_empty_function(T C::*mp) noexcept
    {
        return mp == nullptr;
    }

    inline bool is_empty_function_impl(function_base const* f) noexcept
    {
        return f->empty();
    }

    inline HPX_CONSTEXPR bool is_empty_function_impl(...) noexcept
    {
        return false;
    }

    template <typename F>
    HPX_CONSTEXPR bool is_empty_function(F const& f) noexcept
    {
        return detail::is_empty_function_impl(&f);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig, bool Copyable, bool Serializable>
    class basic_function;

    template <bool Copyable, typename R, typename ...Ts>
    class basic_function<R(Ts...), Copyable, /*Serializable*/false>
      : public function_base
    {
        using base_type = function_base;
        using vtable = function_vtable<R(Ts...), Copyable>;

    public:
        HPX_CONSTEXPR basic_function() noexcept
          : base_type(get_empty_vtable())
        {}

        basic_function(basic_function const& other)
          : base_type(other, get_empty_vtable())
        {}

        basic_function(basic_function&& other) noexcept
          : base_type(std::move(other), get_empty_vtable())
        {}

        basic_function& operator=(basic_function const& other)
        {
            base_type::op_assign(other, get_empty_vtable());
            return *this;
        }

        basic_function& operator=(basic_function&& other) noexcept
        {
            base_type::op_assign(std::move(other), get_empty_vtable());
            return *this;
        }

        void assign(std::nullptr_t) noexcept
        {
            base_type::reset(get_empty_vtable());
        }

        template <typename F>
        void assign(F&& f)
        {
            using T = typename std::decay<F>::type;
            static_assert(!Copyable ||
                std::is_constructible<T, T const&>::value,
                "F shall be CopyConstructible");

            if (!detail::is_empty_function(f))
            {
                vtable const* f_vptr =  get_vtable<T>();
                void* buffer = nullptr;
                if (vptr == f_vptr)
                {
                    HPX_ASSERT(object != nullptr);
                    // reuse object storage
                    buffer = object;
                    vtable::template get<T>(object).~T();
                } else {
                    destroy();
                    vptr = f_vptr;
                    buffer = vtable::template allocate<T>(
                        storage, function_storage_size);
                }
                object = ::new (buffer) T(std::forward<F>(f));
            } else {
                base_type::reset(get_empty_vtable());
            }
        }

        void reset() noexcept
        {
            base_type::reset(get_empty_vtable());
        }

        using base_type::swap;
        using base_type::empty;
        using base_type::operator bool;

        template <typename T>
        T* target() noexcept
        {
            using TD = typename std::remove_cv<T>::type;
            static_assert(
                traits::is_invocable_r<R, TD&, Ts...>::value
              , "T shall be Callable with the function signature");

            vtable const* f_vptr =  get_vtable<TD>();
            if (vptr != f_vptr || empty())
                return nullptr;

            return &vtable::template get<TD>(object);
        }

        template <typename T>
        T const* target() const noexcept
        {
            using TD = typename std::remove_cv<T>::type;
            static_assert(
                traits::is_invocable_r<R, TD&, Ts...>::value
              , "T shall be Callable with the function signature");

            vtable const* f_vptr =  get_vtable<TD>();
            if (vptr != f_vptr || empty())
                return nullptr;

            return &vtable::template get<TD>(object);
        }

        HPX_FORCEINLINE R operator()(Ts... vs) const
        {
            vtable const* vptr = static_cast<vtable const*>(base_type::vptr);
            return vptr->invoke(object, std::forward<Ts>(vs)...);
        }

        using base_type::get_function_address;
        using base_type::get_function_annotation;
        using base_type::get_function_annotation_itt;

    private:
        static HPX_CONSTEXPR vtable const* get_empty_vtable() noexcept
        {
            return detail::get_empty_function_vtable<R(Ts...)>();
        }

        template <typename T>
        static vtable const* get_vtable() noexcept
        {
            return detail::get_vtable<vtable, T>();
        }

    protected:
        using base_type::vptr;
        using base_type::object;
        using base_type::storage;
    };

    template <bool Copyable, typename R, typename ...Ts>
    class basic_function<R(Ts...), Copyable, /*Serializable*/true>
      : public basic_function<R(Ts...), Copyable, /*Serializable*/false>
    {
        using vtable = function_vtable<R(Ts...), Copyable>;
        using serializable_vtable = serializable_function_vtable<vtable>;
        using base_type = basic_function<R(Ts...), Copyable, false>;

    public:
        HPX_CONSTEXPR basic_function() noexcept
          : base_type()
          , serializable_vptr(nullptr)
        {}

        template <typename F>
        void assign(F&& f)
        {
            using target_type = typename std::decay<F>::type;

            base_type::assign(std::forward<F>(f));
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

        void save(serialization::output_archive& ar, unsigned const version) const
        {
            bool const is_empty = base_type::empty();
            ar << is_empty;
            if (!is_empty)
            {
                std::string const name = serializable_vptr->name;
                ar << name;

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
                serializable_vptr = detail::get_serializable_vtable<vtable>(name);

                vptr = serializable_vptr->vptr;
                object = serializable_vptr->load_object(
                    storage, function_storage_size, ar, version);
            }
        }

        HPX_SERIALIZATION_SPLIT_MEMBER()

        template <typename T>
        static serializable_vtable const* get_serializable_vtable() noexcept
        {
            return detail::get_serializable_vtable<vtable, T>();
        }

    protected:
        using base_type::vptr;
        using base_type::object;
        using base_type::storage;
        serializable_vtable const* serializable_vptr;
    };
}}}

#endif
