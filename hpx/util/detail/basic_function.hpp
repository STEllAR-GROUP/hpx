//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
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
#include <hpx/util/detail/empty_function.hpp>
#include <hpx/util/detail/vtable/serializable_function_vtable.hpp>
#include <hpx/util/detail/vtable/serializable_vtable.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

#include <cstddef>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace util { namespace detail
{
    static const std::size_t function_storage_size = 3*sizeof(void*);

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable, typename Sig>
    class function_base;

    template <typename VTable, typename R, typename ...Ts>
    class function_base<VTable, R(Ts...)>
    {
    public:
        function_base() noexcept
          : vptr(detail::get_empty_function_vtable<VTable>())
          , object(nullptr)
        {}

        function_base(function_base&& other) noexcept
          : vptr(other.vptr)
          , object(other.object)
        {
            if (object == &other.storage)
            {
                std::memcpy(storage, other.storage, function_storage_size);
                object = &storage;
            }
            other.vptr = detail::get_empty_function_vtable<VTable>();
            other.object = nullptr;
        }

        ~function_base()
        {
            reset();
        }

        function_base& operator=(function_base&& other) noexcept
        {
            if (this != &other)
            {
                swap(other);
                other.reset();
            }
            return *this;
        }

        void assign(std::nullptr_t) noexcept
        {
            reset();
        }

        template <typename F>
        void assign(F&& f)
        {
            if (!is_empty_function(f))
            {
                typedef typename std::decay<F>::type target_type;

                VTable const* f_vptr = get_vtable<target_type>();
                if (vptr == f_vptr)
                {
                    // reuse object storage
                    vtable::_destruct<target_type>(object);
                    object = vtable::construct<target_type>(
                        object, -1, std::forward<F>(f));
                } else {
                    reset();

                    vptr = f_vptr;
                    object = vtable::construct<target_type>(
                        storage, function_storage_size, std::forward<F>(f));
                }
            } else {
                reset();
            }
        }

        void reset() noexcept
        {
            if (object != nullptr)
            {
                vptr->delete_(object, function_storage_size);

                vptr = detail::get_empty_function_vtable<VTable>();
                object = nullptr;
            }
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

        template <typename T>
        T* target() noexcept
        {
            typedef typename std::remove_cv<T>::type target_type;

            static_assert(
                traits::is_invocable_r<R, target_type&, Ts...>::value
              , "T shall be Callable with the function signature");

            VTable const* f_vptr = get_vtable<target_type>();
            if (vptr != f_vptr || empty())
                return nullptr;

            return &vtable::get<target_type>(object);
        }

        template <typename T>
        T const* target() const noexcept
        {
            typedef typename std::remove_cv<T>::type target_type;

            static_assert(
                traits::is_invocable_r<R, target_type&, Ts...>::value
              , "T shall be Callable with the function signature");

            VTable const* f_vptr = get_vtable<target_type>();
            if (vptr != f_vptr || empty())
                return nullptr;

            return &vtable::get<target_type>(object);
        }

        HPX_FORCEINLINE R operator()(Ts... vs) const
        {
            return vptr->invoke(object, std::forward<Ts>(vs)...);
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
            static util::itt::string_handle sh;
            return sh;
#endif
        }

    private:
        template <typename T>
        static VTable const* get_vtable() noexcept
        {
            return detail::get_vtable<VTable, T>();
        }

    protected:
        VTable const *vptr;
        void* object;
        mutable unsigned char storage[function_storage_size];
    };

    template <typename Sig, typename VTable>
    static bool is_empty_function(function_base<VTable, Sig> const& f) noexcept
    {
        return f.empty();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable, typename Sig, bool Serializable>
    class basic_function;

    template <typename VTable, typename R, typename ...Ts>
    class basic_function<VTable, R(Ts...), true>
      : public function_base<VTable, R(Ts...)>
    {
        using serializable_vtable = serializable_function_vtable<VTable>;
        using base_type = function_base<VTable, R(Ts...)>;

    public:
        typedef R result_type;

        basic_function() noexcept
          : base_type()
          , serializable_vptr(nullptr)
        {}

        basic_function(basic_function&& other) noexcept
          : base_type(static_cast<base_type&&>(other))
          , serializable_vptr(other.serializable_vptr)
        {}

        basic_function& operator=(basic_function&& other) noexcept
        {
            base_type::operator=(static_cast<base_type&&>(other));
            serializable_vptr = other.serializable_vptr;
            return *this;
        }

        template <typename F>
        void assign(F&& f)
        {
            base_type::assign(std::forward<F>(f));
            if (!base_type::empty())
            {
                typedef typename std::decay<F>::type target_type;

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
                serializable_vptr = detail::get_serializable_vtable<VTable>(name);

                vptr = serializable_vptr->vptr;
                object = serializable_vptr->load_object(
                    storage, function_storage_size, ar, version);
            }
        }

        HPX_SERIALIZATION_SPLIT_MEMBER()

        template <typename T>
        static serializable_vtable const* get_serializable_vtable() noexcept
        {
            return detail::get_vtable<serializable_vtable, T>();
        }

    protected:
        void copy_serializable_vptr(basic_function const& other) noexcept
        {
            serializable_vptr = other.serializable_vptr;
        }

    protected:
        using base_type::vptr;
        using base_type::object;
        using base_type::storage;
        serializable_vtable const* serializable_vptr;
    };

    template <typename VTable, typename R, typename ...Ts>
    class basic_function<VTable, R(Ts...), false>
      : public function_base<VTable, R(Ts...)>
    {
        using base_type = function_base<VTable, R(Ts...)>;

    public:
        typedef R result_type;

        basic_function() noexcept
          : base_type()
        {}

        basic_function(basic_function&& other) noexcept
          : base_type(static_cast<base_type&&>(other))
        {}

        basic_function& operator=(basic_function&& other) noexcept
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }

    protected:
        void copy_serializable_vptr(basic_function const&) noexcept
        {}
    };

    template <typename Sig, typename VTable, bool Serializable>
    static bool is_empty_function(
        basic_function<VTable, Sig, Serializable> const& f) noexcept
    {
        return f.empty();
    }
}}}

#endif
