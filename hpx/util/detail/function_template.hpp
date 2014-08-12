//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_FUNCTION_TEMPLATE_HPP
#define HPX_UTIL_DETAIL_FUNCTION_TEMPLATE_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/detail/basic_function.hpp>
#include <hpx/util/detail/function_registration.hpp>
#include <hpx/util/detail/vtable/callable_vtable.hpp>
#include <hpx/util/detail/vtable/copyable_vtable.hpp>
#include <hpx/util/detail/vtable/serializable_vtable.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

#include <utility>

namespace hpx { namespace util { namespace detail
{
    template <typename Function>
    struct init_registration;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig, typename IAr, typename OAr>
    struct function_vtable_ptr;

    template <typename Sig>
    struct function_vtable_ptr<Sig, void, void>
    {
        typename callable_vtable<Sig>::invoke_t invoke;
        copyable_vtable::copy_t copy;
        vtable::get_type_t get_type;
        vtable::destruct_t destruct;
        vtable::delete_t delete_;
        bool empty;

        template <typename T>
        function_vtable_ptr(boost::mpl::identity<T>) BOOST_NOEXCEPT
          : invoke(&callable_vtable<Sig>::template invoke<T>)
          , copy(&copyable_vtable::template copy<T>)
          , get_type(&vtable::template get_type<T>)
          , destruct(&vtable::template destruct<T>)
          , delete_(&vtable::template delete_<T>)
          , empty(boost::is_same<T, empty_function<Sig> >::value)
        {}

        template <typename T, typename Arg>
        BOOST_FORCEINLINE static void construct(void** v, Arg&& arg)
        {
            vtable::construct<T>(v, std::forward<Arg>(arg));
        }

        template <typename T, typename Arg>
        BOOST_FORCEINLINE static void reconstruct(void** v, Arg&& arg)
        {
            vtable::reconstruct<T>(v, std::forward<Arg>(arg));
        }
    };

    template <typename Sig, typename IAr, typename OAr>
    struct function_vtable_ptr
      : function_vtable_ptr<Sig, void, void>
    {
        char const* name;
        typename serializable_vtable<IAr, OAr>::save_object_t save_object;
        typename serializable_vtable<IAr, OAr>::load_object_t load_object;

        template <typename T>
        function_vtable_ptr(boost::mpl::identity<T>) BOOST_NOEXCEPT
          : function_vtable_ptr<Sig, void, void>(boost::mpl::identity<T>())
          , name(get_function_name<std::pair<function_vtable_ptr, T> >())
          , save_object(&serializable_vtable<IAr, OAr>::template save_object<T>)
          , load_object(&serializable_vtable<IAr, OAr>::template load_object<T>)
        {
            init_registration<
                std::pair<function_vtable_ptr, T>
            >::g.register_function();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // registration code for serialization
    template <typename Sig, typename IAr, typename OAr, typename T>
    struct init_registration<
        std::pair<function_vtable_ptr<Sig, IAr, OAr>, T>
    >
    {
        typedef std::pair<function_vtable_ptr<Sig, IAr, OAr>, T> vtable_ptr;

        static automatic_function_registration<vtable_ptr> g;
    };

    template <typename Sig, typename IAr, typename OAr, typename T>
    automatic_function_registration<
        std::pair<function_vtable_ptr<Sig, IAr, OAr>, T>
    > init_registration<
        std::pair<function_vtable_ptr<Sig, IAr, OAr>, T>
    >::g =  automatic_function_registration<
                std::pair<function_vtable_ptr<Sig, IAr, OAr>, T>
            >();
}}}

namespace boost { namespace serialization
{
    template <typename Sig, typename IArchive, typename OArchive>
    struct tracking_level< ::hpx::util::detail::function_vtable_ptr<
        Sig, IArchive, OArchive
    > > : boost::mpl::int_<boost::serialization::track_never>
    {};
}}

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Sig
      , typename IArchive = portable_binary_iarchive
      , typename OArchive = portable_binary_oarchive
    >
    class function
      : public detail::basic_function<
            detail::function_vtable_ptr<Sig, IArchive, OArchive>
          , Sig
        >
    {
        typedef detail::function_vtable_ptr<Sig, IArchive, OArchive> vtable_ptr;
        typedef detail::basic_function<vtable_ptr, Sig> base_type;

    public:
        typedef typename base_type::result_type result_type;

        function() BOOST_NOEXCEPT
          : base_type()
        {}

        function(function const& other)
          : base_type()
        {
            detail::vtable::destruct<detail::empty_function<Sig> >(&this->object);

            this->vptr = other.vptr;
            if (!this->vptr->empty)
            {
                this->vptr->copy(&this->object, &other.object);
            }
        }

        function(function&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}

        template <typename F>
        function(F&& f,
            typename boost::disable_if<
                boost::is_same<function, typename util::decay<F>::type>
            >::type* = 0
        ) : base_type()
        {
            assign(std::forward<F>(f));
        }

        function& operator=(function const& other)
        {
            if (this != &other)
            {
                reset();
                detail::vtable::destruct<detail::empty_function<Sig> >(&this->object);

                this->vptr = other.vptr;
                if (!this->vptr->empty)
                {
                    this->vptr->copy(&this->object, &other.object);
                }
            }
            return *this;
        }

        function& operator=(function&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }

        template <typename F>
        typename boost::disable_if<
            boost::is_same<function, typename util::decay<F>::type>
          , function&
        >::type operator=(F&& f)
        {
            assign(std::forward<F>(f));
            return *this;
        }

        using base_type::operator();
        using base_type::assign;
        using base_type::reset;
        using base_type::empty;
        using base_type::target_type;
        using base_type::target;

    private:
        friend class boost::serialization::access;

        void load(IArchive& ar, const unsigned version)
        {
            reset();

            bool is_empty = false;
            ar.load(is_empty);
            if (!is_empty)
            {
                std::string name;
                ar.load(name);

                this->vptr = detail::get_table_ptr<vtable_ptr>(name);
                this->vptr->load_object(&this->object, ar, version);
            }
        }

        void save(OArchive& ar, const unsigned version) const
        {
            bool is_empty = empty();
            ar.save(is_empty);
            if (!is_empty)
            {
                std::string function_name = this->vptr->name;
                ar.save(function_name);

                this->vptr->save_object(&this->object, ar, version);
            }
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig>
    class function<Sig, void, void>
      : public detail::basic_function<
            detail::function_vtable_ptr<Sig, void, void>
          , Sig
        >
    {
        typedef detail::function_vtable_ptr<Sig, void, void> vtable_ptr;
        typedef detail::basic_function<vtable_ptr, Sig> base_type;

    public:
        typedef typename base_type::result_type result_type;

        function() BOOST_NOEXCEPT
          : base_type()
        {}

        function(function const& other)
          : base_type()
        {
            detail::vtable::destruct<detail::empty_function<Sig> >(&this->object);

            this->vptr = other.vptr;
            if (!this->vptr->empty)
            {
                this->vptr->copy(&this->object, &other.object);
            }
        }

        function(function&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}

        template <typename F>
        function(F&& f,
            typename boost::disable_if<
                boost::is_same<function, typename util::decay<F>::type>
            >::type* = 0
        ) : base_type()
        {
            assign(std::forward<F>(f));
        }

        function& operator=(function const& other)
        {
            if (this != &other)
            {
                reset();
                detail::vtable::destruct<detail::empty_function<Sig> >(&this->object);

                this->vptr = other.vptr;
                if (!this->vptr->empty)
                {
                    this->vptr->copy(&this->object, &other.object);
                }
            }
            return *this;
        }

        function& operator=(function&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }

        template <typename F>
        typename boost::disable_if<
            boost::is_same<function, typename util::decay<F>::type>
          , function&
        >::type operator=(F&& f)
        {
            assign(std::forward<F>(f));
            return *this;
        }

        using base_type::operator();
        using base_type::assign;
        using base_type::reset;
        using base_type::empty;
        using base_type::target_type;
        using base_type::target;
    };

    template <typename Sig, typename IArchive, typename OArchive>
    static bool is_empty_function(function<Sig, IArchive, OArchive> const& f) BOOST_NOEXCEPT
    {
        return f.empty();
    }

    ///////////////////////////////////////////////////////////////////////////
#   ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES

    template <typename Sig>
    using function_nonser = function<Sig, void, void>;

#   else

    template <typename Sig>
    class function_nonser
      : public function<Sig, void, void>
    {
        typedef function<Sig, void, void> base_type;

    public:
        function_nonser() BOOST_NOEXCEPT
          : base_type()
        {}

        function_nonser(function_nonser const& other)
          : base_type(static_cast<base_type const&>(other))
        {}

        function_nonser(function_nonser&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}

        template <typename F>
        function_nonser(F&& f,
            typename boost::disable_if<
                boost::is_same<function_nonser, typename util::decay<F>::type>
            >::type* = 0
        ) : base_type(std::forward<F>(f))
        {}

        function_nonser& operator=(function_nonser const& other)
        {
            base_type::operator=(static_cast<base_type const&>(other));
            return *this;
        }

        function_nonser& operator=(function_nonser&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }

        template <typename F>
        typename boost::disable_if<
            boost::is_same<function_nonser, typename util::decay<F>::type>
          , function_nonser&
        >::type operator=(F&& f)
        {
            base_type::operator=(std::forward<F>(f));
            return *this;
        }
    };

    template <typename Sig>
    static bool is_empty_function(function_nonser<Sig> const& f) BOOST_NOEXCEPT
    {
        return f.empty();
    }

#   endif /*BOOST_NO_CXX11_TEMPLATE_ALIASES*/
}}

#endif
