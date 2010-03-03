//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_MANAGE_OBJECT_ACTION_JAN_26_2010_0141PM)
#define HPX_RUNTIME_ACTIONS_MANAGE_OBJECT_ACTION_JAN_26_2010_0141PM

#include <cstring>
#include <boost/serialization/serialization.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    struct HPX_API_EXPORT manage_object_action_base
    {
        typedef void (*construct_function)(void*, std::size_t);
        typedef void (*clone_function)(void*, void const*, std::size_t);
        typedef void (*destruct_function)(void*);

    private:
        static void construct_(void*, std::size_t) {}
        static void clone_(void* dest, void const* src, std::size_t size)
        {
            using namespace std;    // some systems have memcpy in std
            memcpy(dest, src, size);
        }
        static void destruct_(void*) {}

    public:
        virtual ~manage_object_action_base() {}

        virtual construct_function construct() const 
        { 
            return &manage_object_action_base::construct_; 
        }
        virtual clone_function clone() const 
        { 
            return &manage_object_action_base::clone_; 
        }
        virtual destruct_function destruct() const 
        { 
            return &manage_object_action_base::destruct_; 
        }
        virtual manage_object_action_base const& get_instance() const;

    private:
        // serialization support, just serialize the type
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int) {}
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct manage_object_action : manage_object_action_base        
    {
        manage_object_action() {}
        ~manage_object_action() {}

    private:
        static void construct_(void* memory, std::size_t size)
        {
            BOOST_ASSERT(size == sizeof(T));
            new (memory) T;
        }
        static void clone_(void* dest, void const* src, std::size_t size)
        {
            BOOST_ASSERT(size == sizeof(T));
            new (dest) T (*reinterpret_cast<T const*>(src));
        }
        static void destruct_(void* memory)
        {
            reinterpret_cast<T*>(memory)->~T();
        }

    private:
        construct_function construct() const 
        { 
            return &manage_object_action::construct_; 
        }
        clone_function clone() const
        {
            return &manage_object_action::clone_; 
        }
        destruct_function destruct() const
        {
            return &manage_object_action::destruct_; 
        }

    public:
        manage_object_action const& get_instance() const
        {
            static manage_object_action const instance =
                manage_object_action();
            return instance;
        }

    private:
        // serialization support, just serialize the type
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int) {}
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    struct manage_object_action<boost::uint8_t> : manage_object_action_base        
    {
        manage_object_action() {}

    private:
        // serialization support, just serialize the type
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int) {}
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif


