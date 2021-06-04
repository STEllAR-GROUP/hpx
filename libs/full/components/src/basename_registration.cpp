//  Copyright (c) 2011-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/components/basename_registration.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////
namespace hpx {
    namespace detail {

        std::string name_from_basename(
            std::string const& basename, std::size_t idx)
        {
            HPX_ASSERT(!basename.empty());

            std::string name;
            if (basename[0] != '/')
            {
                name = '/';
            }

            name += basename;
            if (name[name.size() - 1] != '/')
            {
                name += '/';
            }
            name += std::to_string(idx);

            return name;
        }

        std::string name_from_basename(std::string&& basename, std::size_t idx)
        {
            HPX_ASSERT(!basename.empty());

            std::string name;
            if (basename[0] != '/')
            {
                name = '/';
                name += std::move(basename);
            }
            else
            {
                name = std::move(basename);
            }

            if (name[name.size() - 1] != '/')
            {
                name += '/';
            }
            name += std::to_string(idx);

            return name;
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    std::vector<hpx::future<hpx::id_type>> find_all_from_basename(
        std::string basename, std::size_t num_ids)
    {    // -V813
        if (basename.empty())
        {
            HPX_THROW_EXCEPTION(bad_parameter, "hpx::find_all_from_basename",
                "no basename specified");
        }

        std::vector<hpx::future<hpx::id_type>> results;
        for (std::size_t i = 0; i != num_ids; ++i)
        {
            std::string name = detail::name_from_basename(basename, i);
            results.push_back(
                agas::on_symbol_namespace_event(std::move(name), true));
        }
        return results;
    }

    std::vector<hpx::future<hpx::id_type>> find_from_basename(
        std::string basename, std::vector<std::size_t> const& ids)
    {    // -V813
        if (basename.empty())
        {
            HPX_THROW_EXCEPTION(bad_parameter, "hpx::find_from_basename",
                "no basename specified");
        }

        std::vector<hpx::future<hpx::id_type>> results;
        for (std::size_t i : ids)
        {
            std::string name =
                detail::name_from_basename(basename, i);    //-V106
            results.push_back(
                agas::on_symbol_namespace_event(std::move(name), true));
        }
        return results;
    }

    hpx::future<hpx::id_type> find_from_basename(std::string basename,
        std::size_t sequence_nr)    // -V813
    {
        if (basename.empty())
        {
            HPX_THROW_EXCEPTION(bad_parameter, "hpx::find_from_basename",
                "no basename specified");
        }

        if (sequence_nr == ~static_cast<std::size_t>(0))
        {
            sequence_nr = static_cast<std::size_t>(agas::get_locality_id());
        }

        std::string name =
            detail::name_from_basename(std::move(basename), sequence_nr);
        return agas::on_symbol_namespace_event(std::move(name), true);
    }

    hpx::future<bool> register_with_basename(std::string basename,
        hpx::id_type id, std::size_t sequence_nr)    // -V813
    {
        if (basename.empty())
        {
            HPX_THROW_EXCEPTION(bad_parameter, "hpx::register_with_basename",
                "no basename specified");
        }

        if (sequence_nr == ~static_cast<std::size_t>(0))
        {
            sequence_nr = static_cast<std::size_t>(agas::get_locality_id());
        }

        std::string name =
            detail::name_from_basename(std::move(basename), sequence_nr);
        return agas::register_name(std::move(name), id);
    }

    hpx::future<bool> register_with_basename(std::string base_name,
        hpx::future<hpx::id_type> f, std::size_t sequence_nr)
    {
        return f.then(hpx::launch::sync,
            [sequence_nr, base_name = std::move(base_name)](
                hpx::future<hpx::id_type>&& f) mutable -> hpx::future<bool> {
                return register_with_basename(
                    std::move(base_name), f.get(), sequence_nr);
            });
    }

    hpx::future<hpx::id_type> unregister_with_basename(
        std::string basename, std::size_t sequence_nr)
    {    // -V813
        if (basename.empty())
        {
            HPX_THROW_EXCEPTION(bad_parameter, "hpx::unregister_with_basename",
                "no basename specified");
        }

        if (sequence_nr == ~static_cast<std::size_t>(0))
        {
            sequence_nr = static_cast<std::size_t>(agas::get_locality_id());
        }

        std::string name =
            detail::name_from_basename(std::move(basename), sequence_nr);
        return agas::unregister_name(std::move(name));
    }
}    // namespace hpx
