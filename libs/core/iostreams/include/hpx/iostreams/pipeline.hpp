//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#pragma once

#include <hpx/config.hpp>
#include <hpx/iostreams/traits.hpp>

namespace hpx::iostreams {

    HPX_CXX_CORE_EXPORT template <typename Pipeline, typename Component>
    struct pipeline;

    namespace detail {

        HPX_CXX_CORE_EXPORT template <typename Component>
        class pipeline_segment
        {
        public:
            using char_type = char_type_of_t<Component>;
            using category = category_of_t<Component>;

            explicit constexpr pipeline_segment(
                Component const& component) noexcept
              : component_(component)
            {
            }

            template <typename Fn>
            void for_each(Fn fn) const
            {
                fn(component_);
            }

            template <typename Chain>
            void push(Chain& chn) const
            {
                chn.push(component_);
            }

        private:
            Component const& component_;
        };
    }    // namespace detail

    //------------------Definition of Pipeline------------------------------------//
    HPX_CXX_CORE_EXPORT template <typename Pipeline, typename Component>
    struct pipeline : Pipeline
    {
        using pipeline_type = Pipeline;
        using component_type = Component;

        constexpr pipeline(
            Pipeline const& p, Component const& component) noexcept
          : Pipeline(p)
          , component_(component)
        {
        }

        template <typename Fn>
        void for_each(Fn fn) const
        {
            Pipeline::for_each(fn);
            fn(component_);
        }

        template <typename Chain>
        void push(Chain& chn) const
        {
            Pipeline::push(chn);
            chn.push(component_);
        }

        Pipeline const& tail() const
        {
            return *this;
        }

        Component const& head() const
        {
            return component_;
        }

    private:
        Component const& component_;
    };

    HPX_CXX_CORE_EXPORT template <typename Pipeline, typename Filter,
        typename Component>
        requires(is_filter_v<Filter>)
    constexpr pipeline<pipeline<Pipeline, Filter>, Component> operator|(
        pipeline<Pipeline, Filter>& p, Component const& cmp) noexcept
    {
        return pipeline<pipeline<Pipeline, Filter>, Component>(p, cmp);
    }

    HPX_CXX_CORE_EXPORT template <typename Component,
        template <class...> typename Filter, typename... Ts>
        requires(is_filter_v<Filter<Ts...>>)
    constexpr pipeline<detail::pipeline_segment<Filter<Ts...>>, Component>
    operator|(Filter<Ts...> const& f, Component const& c) noexcept
    {
        using segment = detail::pipeline_segment<Filter<Ts...>>;
        return pipeline<segment, Component>(segment(f), c);
    }
}    // namespace hpx::iostreams
