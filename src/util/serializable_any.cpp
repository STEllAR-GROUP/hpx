/*=============================================================================
    Copyright (c) 2013 Shuangyang Yang
    Copyright (c) 2007-2019 Hartmut Kaiser
    Copyright (c) Christopher Diggins 2005
    Copyright (c) Pablo Aguilar 2005
    Copyright (c) Kevlin Henney 2001

//  SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

    The class hpx::util::any is built based on boost::spirit::hold_any class.
    It adds support for HPX serialization, move assignment, == operator.
==============================================================================*/

#include <hpx/config.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/util/serializable_any.hpp>

#include <boost/functional/hash.hpp>

#include <cstddef>
#include <type_traits>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {

    ////////////////////////////////////////////////////////////////////////////
    namespace detail {

        struct hash_binary_filter : serialization::binary_filter
        {
            explicit hash_binary_filter(std::size_t seed = 0)
              : hash(seed)
            {
            }

            // compression API
            void set_max_length(std::size_t size) {}
            void save(void const* src, std::size_t src_count)
            {
                char const* data = static_cast<char const*>(src);
                boost::hash_range(hash, data, data + src_count);
            }
            bool flush(void* dst, std::size_t dst_count, std::size_t& written)
            {
                written = dst_count;
                return true;
            }

            // decompression API
            std::size_t init_data(char const* buffer, std::size_t size,
                std::size_t buffer_size)
            {
                return 0;
            }
            void load(void* dst, std::size_t dst_count) {}

            template <class T>
            void serialize(T&, unsigned)
            {
            }
            HPX_SERIALIZATION_POLYMORPHIC(hash_binary_filter);

            std::size_t hash;
        };
    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////////
    template <typename Char>
    std::size_t hash_any::operator()(
        const basic_any<serialization::input_archive,
            serialization::output_archive, Char, std::true_type>& elem) const
    {
        detail::hash_binary_filter hasher;

        {
            std::vector<char> data;
            serialization::output_archive ar(
                data, 0U, nullptr, &hasher);
            ar << elem;
        }    // let archive go out of scope

        return hasher.hash;
    }

    template HPX_EXPORT std::size_t hash_any::operator()(
        const basic_any<serialization::input_archive,
            serialization::output_archive, char, std::true_type>& elem) const;

    template HPX_EXPORT std::size_t hash_any::operator()(
        const basic_any<serialization::input_archive,
            serialization::output_archive, wchar_t, std::true_type>& elem) const;

}}    // namespace hpx::util

