//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_PARCELSET_POLICIES_COALESCING_MESSAGE_HANDLER_FEB_24_2013_0302PM)
#define HPX_RUNTIME_PARCELSET_POLICIES_COALESCING_MESSAGE_HANDLER_FEB_24_2013_0302PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/policies/message_handler.hpp>
#include <hpx/util/reinitializable_static.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace parcel
{
    struct HPX_LIBRARY_EXPORT coalescing_message_handler
      : parcelset::policies::message_handler
    {
        typedef parcelset::policies::message_handler::write_handler_type
            write_handler_type;

        coalescing_message_handler(std::size_t num);
        ~coalescing_message_handler();

        void put_parcel(parcelset::parcelport* set, parcelset::parcel& p,
            write_handler_type f);

        template <typename Action>
        static parcelset::policies::message_handler* 
            get_message_handler(std::size_t num);

    private:
        std::size_t buffer_size_;
    };

    template <typename Action>
    static parcelset::policies::message_handler* 
        coalescing_message_handler::get_message_handler(std::size_t num)
    {
        util::reinitializable_static<coalescing_message_handler> handler(num);
        return &handler.get();
    }
}}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_ACTION_USES_MESSAGE_COALESCING(action, num)                       \
    namespace hpx { namespace traits                                          \
    {                                                                         \
        template <>                                                           \
        struct action_message_handler<action>                                 \
        {                                                                     \
            static parcelset::policies::message_handler* call()               \
            {                                                                 \
                return coalescing_message_handler::                           \
                    get_message_handler<action>(num);                         \
            }                                                                 \
        };                                                                    \
    }}                                                                        \
/**/

#endif
