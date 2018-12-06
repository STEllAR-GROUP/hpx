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
    template <typename F>
    static bool is_empty_function(F const&, std::false_type) noexcept
    {
        return false;
    }

    template <typename F>
    static bool is_empty_function(F const& f, std::true_type) noexcept
    {
        return f == nullptr;
    }

    template <typename F>
    static bool is_empty_function(F const& f) noexcept
    {
        std::integral_constant<bool,
            std::is_pointer<F>::value
         || std::is_member_pointer<F>::value
        > is_pointer;
        return is_empty_function(f, is_pointer);
    }

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
            reset();

            if (!is_empty_function(f))
            {
                typedef typename std::decay<F>::type target_type;
                vptr = get_vtable<target_type>();
                object = vtable::construct<target_type>(
                    storage, function_storage_size, std::forward<F>(f));
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
      : public function_base<
            serializable_function_vtable<VTable>
          , R(Ts...)
        >
    {
        typedef serializable_function_vtable<VTable> vtable;
        typedef function_base<vtable, R(Ts...)> base_type;

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

    private:
        friend class hpx::serialization::access;

        void load(serialization::input_archive& ar, const unsigned version)
        {
            this->reset();

            bool is_empty = false;
            ar >> is_empty;
            if (!is_empty)
            {
                std::string name;
                ar >> name;

                this->vptr = detail::get_vtable<vtable>(name);
                this->object = this->vptr->load_object(
                    this->storage, function_storage_size, ar, version);
            }
        }

        void save(serialization::output_archive& ar, const unsigned version) const
        {
            bool is_empty = this->empty();
            ar << is_empty;
            if (!is_empty)
            {
                std::string function_name = this->vptr->name;
                ar << function_name;

                this->vptr->save_object(this->object, ar, version);
            }
        }

        HPX_SERIALIZATION_SPLIT_MEMBER()
    };

    template <typename VTable, typename R, typename ...Ts>
    class basic_function<VTable, R(Ts...), false>
      : public function_base<VTable, R(Ts...)>
    {
        typedef function_base<VTable, R(Ts...)> base_type;

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
    };

    template <typename Sig, typename VTable, bool Serializable>
    static bool is_empty_function(
        basic_function<VTable, Sig, Serializable> const& f) noexcept
    {
        return f.empty();
    }
}}}

#endif
