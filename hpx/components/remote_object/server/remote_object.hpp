//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_REMOTE_OBJECT_SERVER_REMOTE_OBJECT_HPP
#define HPX_COMPONENTS_REMOTE_OBJECT_SERVER_REMOTE_OBJECT_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/function.hpp>

namespace hpx { namespace components { namespace server
{
    class HPX_COMPONENT_EXPORT remote_object
        : public simple_component_base<remote_object>
    {
        public:
            remote_object();

            enum actions
            {
                remote_object_apply = 1
            };

            void apply(hpx::util::function<void(void**)> ctor, std::size_t count);

            typedef
                hpx::actions::action2<
                    remote_object
                  , remote_object_apply
                  , hpx::util::function<void(void**)>
                  , std::size_t
                  , &remote_object::apply
                >
                apply_action;
        private:
            void *object;
    };

}}}

#endif
