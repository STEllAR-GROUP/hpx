//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_MANAGE_OBJECT_ACTION_JAN_26_2010_0141PM)
#define HPX_RUNTIME_ACTIONS_MANAGE_OBJECT_ACTION_JAN_26_2010_0141PM

#include <cstring>
#include <boost/config.hpp>
#include <boost/serialization/serialization.hpp>

#include <hpx/config/warnings_prefix.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/util/void_cast.hpp>
#include <hpx/util/reinitializable_static.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    struct HPX_API_EXPORT manage_object_action_base
    {
        typedef void (*construct_function)(void*, std::size_t);
        typedef void (*clone_function)(void*, void const*, std::size_t);
        typedef void (*assign_function)(void*, void const*, std::size_t);
        typedef void (*destruct_function)(void*);

        typedef util::portable_binary_oarchive oarchive_type;
        typedef util::portable_binary_iarchive iarchive_type;

        typedef void (*serialize_save_function)(
            boost::uint8_t const*, std::size_t, oarchive_type&,
            const unsigned int, boost::uint8_t const*);
        typedef void (*serialize_load_function)(
            boost::uint8_t*, std::size_t, iarchive_type&,
            const unsigned int, boost::uint8_t const*);

    private:
        static void construct_(void*, std::size_t) {}
        static void clone_(void* dest, void const* src, std::size_t size)
        {
            using namespace std;    // some systems have memcpy in std
            memcpy(dest, src, size);
        }
        static void assign_(void* dest, void const* src, std::size_t size) //-V524
        {
            using namespace std;    // some systems have memcpy in std
            memcpy(dest, src, size);
        }
        static void destruct_(void*) {}

        static void save_(boost::uint8_t const* data, std::size_t size,
            oarchive_type& ar, const unsigned int,
            boost::uint8_t const*)
        {
            using boost::serialization::make_array;
            ar << make_array(data, size);
        }
        static void load_(boost::uint8_t* data, std::size_t size,
            iarchive_type& ar, const unsigned int,
            boost::uint8_t const*)
        {
            using boost::serialization::make_array;
            ar >> make_array(data, size);
        }

    public:
        manage_object_action_base() {}
        virtual ~manage_object_action_base() {}

        // support for construction, copying, destruction
        virtual construct_function construct() const
        {
            return &manage_object_action_base::construct_;
        }
        virtual clone_function clone() const
        {
            return &manage_object_action_base::clone_;
        }
        virtual assign_function assign() const
        {
            return &manage_object_action_base::assign_;
        }
        virtual destruct_function destruct() const
        {
            return &manage_object_action_base::destruct_;
        }

        struct tag {};

        virtual manage_object_action_base const& get_instance() const;

        // serialization support
        virtual serialize_save_function save() const
        {
            return &manage_object_action_base::save_;
        }
        virtual serialize_load_function load() const
        {
            return &manage_object_action_base::load_;
        }

    private:
        // serialization support, just serialize the type
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive&, const unsigned int) {}
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Config = void>
    struct manage_object_action;

    template <typename T>
    struct manage_object_action<T, void> : manage_object_action_base
    {
        manage_object_action() {}
        ~manage_object_action() {}

    private:
#if defined(NDEBUG) && defined(BOOST_DISABLE_ASSERTS)
        static void construct_(void* memory, std::size_t)
#else
        static void construct_(void* memory, std::size_t size)
#endif
        {
            BOOST_ASSERT(size == sizeof(T));
            new (memory) T;
        }
#if defined(NDEBUG) && defined(BOOST_DISABLE_ASSERTS)
        static void clone_(void* dest, void const* src, std::size_t)
#else
        static void clone_(void* dest, void const* src, std::size_t size)
#endif
        {
            BOOST_ASSERT(size == sizeof(T));
            new (dest) T (*reinterpret_cast<T const*>(src));
        }
#if defined(NDEBUG) && defined(BOOST_DISABLE_ASSERTS)
        static void assign_(void* dest, void const* src, std::size_t)
#else
        static void assign_(void* dest, void const* src, std::size_t size)
#endif
        {
            BOOST_ASSERT(size == sizeof(T));
            // do not overwrite ourselves
            if (src != dest)
                *reinterpret_cast<T*>(dest) = *reinterpret_cast<T const*>(src);
        }
        static void destruct_(void* memory)
        {
            reinterpret_cast<T*>(memory)->~T();
        }

        static void save_(boost::uint8_t const* data, std::size_t /*size*/,
            oarchive_type& ar, const unsigned int /*version*/,
            boost::uint8_t const* /*config*/)
        {
            ar << *reinterpret_cast<T const*>(data);
        }
        static void load_(boost::uint8_t* data, std::size_t /*size*/,
            iarchive_type& ar, const unsigned int /*version*/,
            boost::uint8_t const* /*config*/)
        {
            ar >> *reinterpret_cast<T*>(data);
        }

    private:
        // support for construction, copying, destruction
        construct_function construct() const
        {
            return &manage_object_action::construct_;
        }
        clone_function clone() const
        {
            return &manage_object_action::clone_;
        }
        assign_function assign() const
        {
            return &manage_object_action::assign_;
        }
        destruct_function destruct() const
        {
            return &manage_object_action::destruct_;
        }

        // serialization support
        serialize_save_function save() const
        {
            return &manage_object_action::save_;
        }
        serialize_load_function load() const
        {
            return &manage_object_action::load_;
        }

    public:
        struct tag {};

        virtual manage_object_action_base const& get_instance() const
        {
            // ensure thread-safe initialization
            util::reinitializable_static<manage_object_action, tag> instance;
            return instance.get();
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<manage_object_action, manage_object_action_base>();
        }

    private:
        // serialization support, just serialize the type
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & util::base_object_nonvirt<manage_object_action_base>(*this);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    struct manage_object_action<boost::uint8_t> : manage_object_action_base
    {
        manage_object_action() {}

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<manage_object_action, manage_object_action_base>();
        }

    private:
        // serialization support, just serialize the type
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & util::base_object_nonvirt<manage_object_action_base>(*this);
        }
    };

    inline manage_object_action_base const&
    manage_object_action_base::get_instance() const
    {
        // ensure thread-safe initialization
        util::reinitializable_static<manage_object_action<boost::uint8_t>,
            manage_object_action_base::tag> instance;
        return instance.get();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Config>
    struct manage_object_action : manage_object_action<T>
    {
        manage_object_action() {}
        ~manage_object_action() {}

        typedef typename manage_object_action<T>::oarchive_type oarchive_type;
        typedef typename manage_object_action<T>::iarchive_type iarchive_type;

        typedef typename manage_object_action<T>::serialize_save_function
            serialize_save_function;
        typedef typename manage_object_action<T>::serialize_load_function
            serialize_load_function;

    private:
        static void save_(boost::uint8_t const* data, std::size_t /*size*/,
            oarchive_type& ar, const unsigned int version,
            boost::uint8_t const* config)
        {
            reinterpret_cast<T const*>(data)->save(ar, version,
                reinterpret_cast<Config const*>(config));
        }
        static void load_(boost::uint8_t* data, std::size_t /*size*/,
            iarchive_type& ar, const unsigned int version,
            boost::uint8_t const* config)
        {
            reinterpret_cast<T*>(data)->load(ar, version,
                reinterpret_cast<Config const*>(config));
        }

    private:
        // serialization support
        serialize_save_function save() const
        {
            return &manage_object_action::save_;
        }
        serialize_load_function load() const
        {
            return &manage_object_action::load_;
        }

    public:
        struct tag {};

        manage_object_action_base const& get_instance() const
        {
            // ensure thread-safe initialization
            util::reinitializable_static<manage_object_action, tag> instance;
            return instance.get();
        }

        /// serialization support
        static void register_base()
        {
            util::void_cast_register_nonvirt<manage_object_action, manage_object_action<T> >();
            manage_object_action<T>::register_base();
        }

    private:
        // serialization support, just serialize the type
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & util::base_object_nonvirt<manage_object_action<T> >(*this);
        }
    };
}}

#define HPX_REGISTER_MANAGE_OBJECT_ACTION(object_action, name)                \
        BOOST_CLASS_EXPORT(object_action)                                     \
        HPX_REGISTER_BASE_HELPER(object_action, name)                         \
    /***/

#include <hpx/config/warnings_suffix.hpp>

#endif


