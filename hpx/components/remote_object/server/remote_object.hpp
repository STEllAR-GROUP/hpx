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
    // component to hold a pointer to an object, ability to apply arbitrary
    // functions objects on that pointer.
    class HPX_COMPONENT_EXPORT remote_object
        : public simple_component_base<remote_object>
    {
        public:
            remote_object()
                : object(0)
            {}
            ~remote_object()
            {
                BOOST_ASSERT(!dtor.empty());
                dtor(&object);
            }

            enum actions
            {
                remote_object_apply    = 1
              , remote_object_set_dtor = 2
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
            
            void set_dtor(hpx::util::function<void(void**)> d, std::size_t count);

            typedef
                hpx::actions::action2<
                    remote_object
                  , remote_object_set_dtor
                  , hpx::util::function<void(void**)>
                  , std::size_t
                  , &remote_object::set_dtor
                >
                set_dtor_action;
        private:
            void *object;
            hpx::util::function<void(void**)> dtor;
    };

}}}

#endif
