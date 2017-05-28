//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
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
#include <typeinfo>
#include <utility>

namespace hpx { namespace util { namespace detail
{
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
        HPX_MOVABLE_ONLY(function_base);

        // make sure the empty table instance is initialized in time, even
        // during early startup
        static VTable const* get_empty_table()
        {
            static VTable const empty_table =
                detail::construct_vtable<detail::empty_function<R(Ts...)> >();
            return &empty_table;
        }

    public:
        function_base() noexcept
          : vptr(get_empty_table())
        {
            std::memset(object, 0, vtable::function_storage_size);
            vtable::default_construct<empty_function<R(Ts...)> >(object);
        }

        function_base(function_base&& other) noexcept
          : vptr(other.vptr)
        {
            // move-construct
            std::memcpy(object, other.object, vtable::function_storage_size);
            other.vptr = get_empty_table();
            vtable::default_construct<empty_function<R(Ts...)> >(other.object);
        }

        ~function_base()
        {
            vptr->delete_(object);
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
                    vtable::reconstruct<target_type>(object, std::forward<F>(f));
                } else {
                    reset();
                    vtable::_delete<empty_function<R(Ts...)> >(object);

                    vptr = f_vptr;
                    vtable::construct<target_type>(object, std::forward<F>(f));
                }
            } else {
                reset();
            }
        }

        void reset() noexcept
        {
            if (!vptr->empty)
            {
                vptr->delete_(object);

                vptr = get_empty_table();
                vtable::default_construct<empty_function<R(Ts...)> >(object);
            }
        }

        void swap(function_base& f) noexcept
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object); // swap
        }

        bool empty() const noexcept
        {
            return vptr->empty;
        }

        explicit operator bool() const noexcept
        {
            return !empty();
        }

        std::type_info const& target_type() const noexcept
        {
            return empty() ? typeid(void) : vptr->get_type();
        }

        template <typename T>
        T* target() noexcept
        {
            typedef typename std::remove_cv<T>::type target_type;

            static_assert(
                traits::is_callable<target_type&(Ts...), R>::value
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
                traits::is_callable<target_type&(Ts...), R>::value
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
            return vptr->get_function_address(object);
        }

        char const* get_function_annotation() const
        {
            return vptr->get_function_annotation(object);
        }

    private:
        template <typename T>
        static VTable const* get_vtable() noexcept
        {
            return detail::get_vtable<VTable, T>();
        }

    protected:
        VTable const *vptr;
        mutable void* object[(vtable::function_storage_size / sizeof(void*))];
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
        HPX_MOVABLE_ONLY(basic_function);

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
                this->vptr->load_object(this->object, ar, version);
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
        HPX_MOVABLE_ONLY(basic_function);

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
