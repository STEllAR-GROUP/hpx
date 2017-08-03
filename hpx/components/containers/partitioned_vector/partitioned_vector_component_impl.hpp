//  Copyright (c) 2014 Anuj R. Sharma
//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARTITIONED_VECTOR_COMPONENT_IMPL_HPP
#define HPX_PARTITIONED_VECTOR_COMPONENT_IMPL_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/components/server/component_base.hpp>
#include <hpx/runtime/components/server/component.hpp>
#include <hpx/runtime/get_ptr.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/detail/pp/cat.hpp>
#include <hpx/util/detail/pp/expand.hpp>
#include <hpx/util/detail/pp/nargs.hpp>

#include <hpx/components/containers/partitioned_vector/partitioned_vector_fwd.hpp>

#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace hpx { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Data>
    inline partitioned_vector<T, Data>::partitioned_vector()
    {
        HPX_ASSERT(false);    // shouldn't ever be called
    }

    template <typename T, typename Data>
    inline partitioned_vector<T, Data>::partitioned_vector(
        size_type partition_size)
      : partitioned_vector_partition_(partition_size)
    {
    }

    template <typename T, typename Data>
    inline partitioned_vector<T, Data>::partitioned_vector(
            size_type partition_size, T const& val)
      : partitioned_vector_partition_(partition_size, val)
    {
    }

    template <typename T, typename Data>
    inline partitioned_vector<T, Data>::partitioned_vector(
        size_type partition_size, T const& val, allocator_type const& alloc)
      : partitioned_vector_partition_(partition_size, val, alloc)
    {
    }

    template <typename T, typename Data>
    inline partitioned_vector<T, Data>::partitioned_vector(
        partitioned_vector const& rhs)
      : base_type(rhs)
      , partitioned_vector_partition_(rhs.partitioned_vector_partition_)
    {
    }

    template <typename T, typename Data>
    inline partitioned_vector<T, Data>::partitioned_vector(
        partitioned_vector&& rhs)
      : base_type(std::move(rhs))
      , partitioned_vector_partition_(
            std::move(rhs.partitioned_vector_partition_))
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Data>
    inline typename partitioned_vector<T, Data>::data_type&
    partitioned_vector<T, Data>::get_data()
    {
        return partitioned_vector_partition_;
    }

    template <typename T, typename Data>
    inline typename partitioned_vector<T, Data>::data_type const&
    partitioned_vector<T, Data>::get_data() const
    {
        return partitioned_vector_partition_;
    }

    template <typename T,typename Data>
    inline typename partitioned_vector<T, Data>::data_type
    partitioned_vector<T,Data>::get_copied_data() const
    {
        return partitioned_vector_partition_;
    }

    template <typename T,typename Data>
    void partitioned_vector<T,Data>::set_data(data_type && other)
    {
        partitioned_vector_partition_ = std::move(other);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Data>
    inline typename partitioned_vector<T, Data>::iterator_type
    partitioned_vector<T, Data>::begin()
    {
        return partitioned_vector_partition_.begin();
    }

    template <typename T, typename Data>
    inline typename partitioned_vector<T, Data>::const_iterator_type
    partitioned_vector<T, Data>::begin() const
    {
        return partitioned_vector_partition_.begin();
    }

    template <typename T, typename Data>
    inline typename partitioned_vector<T, Data>::const_iterator_type
    partitioned_vector<T, Data>::cbegin() const
    {
        return partitioned_vector_partition_.cbegin();
    }

    template <typename T, typename Data>
    inline typename partitioned_vector<T, Data>::iterator_type
    partitioned_vector<T, Data>::end()
    {
        return partitioned_vector_partition_.end();
    }

    template <typename T, typename Data>
    inline typename partitioned_vector<T, Data>::const_iterator_type
    partitioned_vector<T, Data>::end() const
    {
        return partitioned_vector_partition_.end();
    }

    template <typename T, typename Data>
    inline typename partitioned_vector<T, Data>::const_iterator_type
    partitioned_vector<T, Data>::cend() const
    {
        return partitioned_vector_partition_.cend();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Data>
    inline typename partitioned_vector<T, Data>::size_type
    partitioned_vector<T, Data>::size() const
    {
        return partitioned_vector_partition_.size();
    }

    template <typename T, typename Data>
    inline typename partitioned_vector<T, Data>::size_type
    partitioned_vector<T, Data>::max_size() const
    {
        return partitioned_vector_partition_.max_size();
    }

    template <typename T, typename Data>
    inline typename partitioned_vector<T, Data>::size_type
    partitioned_vector<T, Data>::capacity() const
    {
        return partitioned_vector_partition_.capacity();
    }

    template <typename T, typename Data>
    inline bool partitioned_vector<T, Data>::empty() const
    {
        return partitioned_vector_partition_.empty();
    }

    template <typename T, typename Data>
    inline void partitioned_vector<T, Data>::resize(
        size_type n, T const& val)
    {
        partitioned_vector_partition_.resize(n, val);
    }

    template <typename T, typename Data>
    inline void partitioned_vector<T, Data>::reserve(size_type n)
    {
        partitioned_vector_partition_.reserve(n);
    }

    template <typename T, typename Data>
    inline T partitioned_vector<T, Data>::get_value(size_type pos) const
    {
        return partitioned_vector_partition_[pos];
    }

    template <typename T, typename Data>
    inline std::vector<T> partitioned_vector<T, Data>::get_values(
        std::vector<size_type> const& pos) const
    {
        std::vector<T> result;
        result.reserve(pos.size());

        for (std::size_t i = 0; i != pos.size(); ++i)
            result.push_back(partitioned_vector_partition_[pos[i]]);

        return result;
    }

    template <typename T, typename Data>
    inline T partitioned_vector<T, Data>::front() const
    {
        return partitioned_vector_partition_.front();
    }

    template <typename T, typename Data>
    inline T partitioned_vector<T, Data>::back() const
    {
        return partitioned_vector_partition_.back();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Data>
    inline void partitioned_vector<T, Data>::assign(
        size_type n, T const& val)
    {
        partitioned_vector_partition_.assign(n, val);
    }

    template <typename T, typename Data>
    inline void partitioned_vector<T, Data>::push_back(T const& val)
    {
        partitioned_vector_partition_.push_back(val);
    }

    template <typename T, typename Data>
    inline void partitioned_vector<T, Data>::pop_back()
    {
        partitioned_vector_partition_.pop_back();
    }

    template <typename T, typename Data>
    inline void partitioned_vector<T, Data>::set_value(
        size_type pos, T const& val)
    {
        partitioned_vector_partition_[pos] = val;
    }

    template <typename T, typename Data>
    inline void partitioned_vector<T, Data>::set_values(
        std::vector<size_type> const& pos, std::vector<T> const& val)
    {
        HPX_ASSERT(pos.size() == val.size());
        HPX_ASSERT(pos.size() <= partitioned_vector_partition_.size());

        for (std::size_t i = 0; i != pos.size(); ++i)
            partitioned_vector_partition_[pos[i]] = val[i];
    }

    template <typename T, typename Data>
    inline void partitioned_vector<T, Data>::clear()
    {
        partitioned_vector_partition_.clear();
    }
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_PARTITIONED_VECTOR(...)                                  \
    HPX_REGISTER_VECTOR_(__VA_ARGS__)                                         \
/**/
#define HPX_REGISTER_VECTOR_(...)                                             \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_REGISTER_VECTOR_, HPX_PP_NARGS(__VA_ARGS__)                       \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_VECTOR_IMPL(type, name)                                  \
    HPX_REGISTER_ACTION(type::get_value_action,                               \
        HPX_PP_CAT(__vector_get_value_action_, name));                        \
    HPX_REGISTER_ACTION(type::get_values_action,                              \
        HPX_PP_CAT(__vector_get_values_action_, name));                       \
    HPX_REGISTER_ACTION(type::set_value_action,                               \
        HPX_PP_CAT(__vector_set_value_action_, name));                        \
    HPX_REGISTER_ACTION(type::set_values_action,                              \
        HPX_PP_CAT(__vector_set_values_action_, name));                       \
    HPX_REGISTER_ACTION(type::size_action,                                    \
        HPX_PP_CAT(__vector_size_action_, name));                             \
    HPX_REGISTER_ACTION(type::resize_action,                                  \
        HPX_PP_CAT(__vector_resize_action_, name));                           \
    HPX_REGISTER_ACTION(type::get_copied_data_action,                         \
        HPX_PP_CAT(__vector_get_copied_data_action_, name));                  \
    HPX_REGISTER_ACTION(type::set_data_action,                                \
        HPX_PP_CAT(__vector_set_data_action_, name));                         \
    typedef ::hpx::components::component<type> HPX_PP_CAT(__vector_, name);   \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(__vector_, name))                       \
/**/

#define HPX_REGISTER_VECTOR_1(type)                                           \
    HPX_REGISTER_VECTOR_2(type, std::vector<type>)                            \
/**/
#define HPX_REGISTER_VECTOR_2(type, data)                                     \
    HPX_REGISTER_VECTOR_3(type, data, type)                                   \
/**/
#define HPX_REGISTER_VECTOR_3(type, data, name)                               \
    typedef ::hpx::server::partitioned_vector<type, data>                     \
        HPX_PP_CAT(__partitioned_vector_, HPX_PP_CAT(type, name));            \
    HPX_REGISTER_VECTOR_IMPL(                                                 \
        HPX_PP_CAT(__partitioned_vector_, HPX_PP_CAT(type, name)), name)      \
/**/

namespace hpx
{
    template <typename T, typename Data /*= std::vector<T> */>
    inline partitioned_vector_partition<T, Data>::partitioned_vector_partition(
        id_type const& gid)
      : base_type(gid)
    {
    }

    template <typename T, typename Data /*= std::vector<T> */>
    inline partitioned_vector_partition<T, Data>::partitioned_vector_partition(
        hpx::shared_future<id_type> const& gid)
      : base_type(gid)
    {
    }

    template <typename T, typename Data /*= std::vector<T> */>
    inline std::shared_ptr<hpx::server::partitioned_vector<T, Data>>
    partitioned_vector_partition<T, Data>::get_ptr() const
    {
        error_code ec(lightweight);
        return hpx::get_ptr<server::partitioned_vector<T, Data>>(this->get_id())
            .get(ec);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    inline hpx::future<std::size_t>
    partitioned_vector_partition<T, Data>::size_async() const
    {
        HPX_ASSERT(this->get_id());
        return hpx::async<typename server_type::size_action>(this->get_id());
    }

    template <typename T, typename Data /*= std::vector<T> */>
    inline std::size_t partitioned_vector_partition<T, Data>::size() const
    {
        return size_async().get();
    }

    template <typename T, typename Data /*= std::vector<T> */>
    inline void partitioned_vector_partition<T, Data>::resize(
        std::size_t n, T const& val /*= T()*/)
    {
        return resize_async(n, val).get();
    }

    template <typename T, typename Data /*= std::vector<T> */>
    inline hpx::future<void> partitioned_vector_partition<T, Data>::resize_async(
        std::size_t n, T const& val /*= T()*/)
    {
        HPX_ASSERT(this->get_id());
        return hpx::async<typename server_type::resize_action>(
            this->get_id(), n, val);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    inline T partitioned_vector_partition<T, Data>::get_value(
        launch::sync_policy, std::size_t pos) const
    {
        return get_value(pos).get();
    }

#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
    template <typename T, typename Data /*= std::vector<T> */>
    inline T partitioned_vector_partition<T, Data>::get_value_sync(
        std::size_t pos) const
    {
        return get_value(launch::sync, pos);
    }
#endif

    template <typename T, typename Data /*= std::vector<T> */>
    inline hpx::future<T> partitioned_vector_partition<T, Data>::get_value(
        std::size_t pos) const
    {
        HPX_ASSERT(this->get_id());
        return hpx::async<typename server_type::get_value_action>(
            this->get_id(), pos);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    inline std::vector<T>
    partitioned_vector_partition<T, Data>::get_values(
        launch::sync_policy, std::vector<std::size_t> const& pos) const
    {
        return get_values(pos).get();
    }

#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
    template <typename T, typename Data /*= std::vector<T> */>
    inline std::vector<T>
    partitioned_vector_partition<T, Data>::get_values_sync(
        std::vector<std::size_t> const& pos) const
    {
        return get_values(launch::sync, pos);
    }
#endif

    template <typename T, typename Data /*= std::vector<T> */>
    inline hpx::future<std::vector<T>>
    partitioned_vector_partition<T, Data>::get_values(
        std::vector<std::size_t> const& pos) const
    {
        HPX_ASSERT(this->get_id());
        return hpx::async<typename server_type::get_values_action>(
            this->get_id(), pos);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    inline void partitioned_vector_partition<T, Data>::set_value(
        launch::sync_policy, std::size_t pos, T && val)
    {
        set_value(pos, std::move(val)).get();
    }

    template <typename T, typename Data /*= std::vector<T> */>
    inline void partitioned_vector_partition<T, Data>::set_value(
        launch::sync_policy, std::size_t pos, T const& val)
    {
        set_value(pos, val).get();
    }

#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
    template <typename T, typename Data /*= std::vector<T> */>
    inline void partitioned_vector_partition<T, Data>::set_value_sync(
        std::size_t pos, T_&& val)
    {
        set_value(launch::sync, pos, std::forward<T_>(val));
    }
#endif

    template <typename T, typename Data /*= std::vector<T> */>
    inline hpx::future<void> partitioned_vector_partition<T, Data>::set_value(
        std::size_t pos, T && val)
    {
        HPX_ASSERT(this->get_id());
        return hpx::async<typename server_type::set_value_action>(
            this->get_id(), pos, std::move(val));
    }

    template <typename T, typename Data /*= std::vector<T> */>
    inline hpx::future<void> partitioned_vector_partition<T, Data>::set_value(
        std::size_t pos, T const& val)
    {
        HPX_ASSERT(this->get_id());
        return hpx::async<typename server_type::set_value_action>(
            this->get_id(), pos, val);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    inline void partitioned_vector_partition<T, Data>::set_values(
        launch::sync_policy, std::vector<std::size_t> const& pos,
        std::vector<T> const& val)
    {
        set_values(pos, val).get();
    }

#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
    template <typename T, typename Data /*= std::vector<T> */>
    inline void partitioned_vector_partition<T, Data>::set_values_sync(
        std::vector<std::size_t> const& pos, std::vector<T> const& val)
    {
        set_values(launch::sync, pos, val);
    }
#endif

    template <typename T, typename Data /*= std::vector<T> */>
    inline hpx::future<void> partitioned_vector_partition<T, Data>::set_values(
        std::vector<std::size_t> const& pos, std::vector<T> const& val)
    {
        HPX_ASSERT(this->get_id());
        return hpx::async<typename server_type::set_values_action>(
            this->get_id(), pos, val);
    }

    template <typename T, typename Data /*= std::vector<T> */>
    inline typename partitioned_vector_partition<T, Data>::server_type::data_type
        partitioned_vector_partition<T, Data>::get_copied_data(
            launch::sync_policy) const
    {
        return get_copied_data().get();
    }

#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
    template <typename T, typename Data /*= std::vector<T> */>
    inline typename partitioned_vector_partition<T, Data>::server_type::data_type
    partitioned_vector_partition<T, Data>::get_copied_data_sync() const
    {
        return get_copied_data(launch::sync);
    }
#endif

    template <typename T, typename Data /*= std::vector<T> */>
    inline hpx::future<
        typename partitioned_vector_partition<T, Data>::server_type::data_type>
    partitioned_vector_partition<T, Data>::get_copied_data() const
    {
        HPX_ASSERT(this->get_id());
        return hpx::async<typename server_type::get_copied_data_action>(
            this->get_id());
    }

    template <typename T, typename Data /*= std::vector<T> */>
    inline void partitioned_vector_partition<T, Data>::set_data(
        launch::sync_policy, typename server_type::data_type&& other) const
    {
        set_data(std::move(other)).get();
    }

#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
    template <typename T, typename Data /*= std::vector<T> */>
    inline void partitioned_vector_partition<T, Data>::set_data_sync(
        typename server_type::data_type&& other) const
    {
        set_data(launch::sync, std::move(other));
    }
#endif

    template <typename T, typename Data /*= std::vector<T> */>
    inline hpx::future<void> partitioned_vector_partition<T, Data>::set_data(
        typename server_type::data_type&& other) const
    {
        HPX_ASSERT(this->get_id());
        return hpx::async<typename server_type::set_data_action>(
            this->get_id(), std::move(other));
    }
}

#endif
