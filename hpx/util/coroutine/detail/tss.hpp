//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This code has been partially adopted from the Boost.Threads library
//
// (C) Copyright 2008 Anthony Williams
// (C) Copyright 2011-2012 Vicente J. Botet Escriba

#if !defined(HPX_UTIL_COROUTINE_DETAIL_TSS_MAR_20_2014_0941AM)
#define HPX_UTIL_COROUTINE_DETAIL_TSS_MAR_20_2014_0941AM

#include <hpx/config.hpp>
#include <hpx/util/coroutine/exception.hpp>
#include <hpx/util/move.hpp>

#include <boost/shared_ptr.hpp>

#include <map>

namespace hpx { namespace util { namespace coroutines
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////////
        struct tss_cleanup_function
        {
            virtual ~tss_cleanup_function() {}

            virtual void operator()(void* data) = 0;
        };

        ///////////////////////////////////////////////////////////////////////
        struct tss_data_node
        {
        private:
            boost::shared_ptr<tss_cleanup_function> func_;
            void* value_;

        public:
            tss_data_node()
              : value_(0)
            {}

            tss_data_node(void* val)
              : func_(),
                value_(val)
            {}

            tss_data_node(boost::shared_ptr<tss_cleanup_function> f, void* val)
              : func_(f),
                value_(val)
            {}

            tss_data_node(tss_data_node&& rhs)
              : func_(std::move(rhs.func_)),
                value_(rhs.value_)
            {
                rhs.func_.reset();
                rhs.value_ = 0;
            }

            ~tss_data_node()
            {
                cleanup();
            }

            tss_data_node& operator=(tss_data_node&& rhs)
            {
                func_ = std::move(rhs.func_);
                value_ = rhs.value_;

                rhs.func_.reset();
                rhs.value_ = 0;
                return *this;
            }

            template <typename T>
            T get_data() const
            {
                HPX_ASSERT(value_ != 0);
                return *reinterpret_cast<T*>(value_);
            }

            template <typename T>
            void set_data(T const& val)
            {
                if (value_ == 0)
                    value_ = new T(val);
                else
                    *reinterpret_cast<T*>(value_) = val;
            }

            void cleanup(bool cleanup_existing = true);

            void reinit(boost::shared_ptr<tss_cleanup_function> const& f,
                void* data, bool cleanup_existing)
            {
                cleanup(cleanup_existing);
                func_ = f;
                value_ = data;
            }

            void* get_value() const
            {
                return value_;
            }

            HPX_MOVABLE_BUT_NOT_COPYABLE(tss_data_node)
        };

        ///////////////////////////////////////////////////////////////////////
        class tss_storage
        {
        private:
            typedef std::map<void const*, tss_data_node> tss_node_data_map;

            tss_data_node const* find_entry(void const* key) const
            {
                tss_node_data_map::const_iterator it = data_.find(key);
                if (it == data_.end())
                    return 0;
                return &(it->second);
            }
            tss_data_node* find_entry(void const* key)
            {
                tss_node_data_map::iterator it = data_.find(key);
                if (it == data_.end())
                    return 0;
                return &(it->second);
            }

        public:
            tss_storage()
            {
            }

            ~tss_storage()
            {
            }

            std::size_t get_thread_data() const
            {
                return 0;
            }
            std::size_t set_thread_data(std::size_t val)
            {
                return 0;
            }

            tss_data_node* find(void const* key)
            {
                tss_node_data_map::iterator current_node = data_.find(key);
                if (current_node != data_.end())
                    return &current_node->second;
                return NULL;
            }

            void insert(void const* key,
                boost::shared_ptr<tss_cleanup_function> const& func, void* tss_data)
            {
                data_.insert(std::make_pair(key, tss_data_node(func, tss_data)));
            }

            void insert(void const* key, void* tss_data)
            {
                boost::shared_ptr<tss_cleanup_function> func;
                insert(key, func, tss_data);
            }

            void erase(void const* key, bool cleanup_existing)
            {
                tss_data_node* node = find(key);
                if (node)
                {
                    if (!cleanup_existing)
                        node->cleanup(false);
                    data_.erase(key);
                }
            }

        private:
            tss_node_data_map data_;
        };

        ///////////////////////////////////////////////////////////////////////
        HPX_EXPORT tss_data_node* find_tss_data(void const* key);

        HPX_EXPORT void* get_tss_data(void const* key);

        HPX_EXPORT void add_new_tss_node(void const* key,
            boost::shared_ptr<tss_cleanup_function> const& func, void* tss_data);

        HPX_EXPORT void erase_tss_node(void const* key,
            bool cleanup_existing = false);

        HPX_EXPORT void set_tss_data(void const* key,
            boost::shared_ptr<tss_cleanup_function> const& func, void* tss_data = 0,
            bool cleanup_existing = false);

        ///////////////////////////////////////////////////////////////////////
        class tss_storage;

        HPX_EXPORT tss_storage* create_tss_storage();
        HPX_EXPORT void delete_tss_storage(tss_storage*& storage);

        HPX_EXPORT std::size_t get_tss_thread_data(tss_storage* storage);
        HPX_EXPORT std::size_t set_tss_thread_data(tss_storage* storage, std::size_t);
    }

    ///////////////////////////////////////////////////////////////////////////
    struct null_thread_id_exception : exception_base {};
}}}

#endif


