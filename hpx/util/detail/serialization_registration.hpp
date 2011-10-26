//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#ifndef HPX_FUNCTION_DETAILSERIALIZATION_REGISTRATION_HPP
#define HPX_FUNCTION_DETAILSERIALIZATION_REGISTRATION_HPP

#include <boost/serialization/export.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <boost/uuid/sha1.hpp>
#include <typeinfo>

namespace hpx { namespace util { namespace detail {

    template <typename T>
    struct guid_initializer_helper
        //: boost::archive::detail::extra_detail::guid_initializer<T>
    {
        typedef boost::archive::detail::extra_detail::guid_initializer<T> base_type;
        base_type guid_init;

        guid_initializer_helper()
            : guid_init()
        {
            T::register_base();
        }

        void initialize() const {}

        guid_initializer_helper const & export_guid() const {
            guid_init.export_guid();
            return *this;
        }
    };

    template <typename T>
    const char * type_hash()
    {
        static char buf[20];
        /// FIXME: this is not portable across different compilers
        static const char * name = typeid(T).name();
        // create a sha1 hash from the string returned by typeid::name
        boost::uuids::detail::sha1 hash;
        hash.process_block(name, name + std::strlen(name));
        unsigned digest[5];
        hash.get_digest(digest);

        // copy that into our string
        std::memcpy(buf, digest, 5*sizeof(unsigned));

        return buf;
    }
}}}

// registering for serialization
namespace boost {
    namespace serialization {
        template <
            typename Sig, typename IArchive, typename OArchive
          , typename Vtable
        >
        struct guid_defined<
            ::hpx::util::detail::vtable_ptr<
                Sig
              , IArchive
              , OArchive
              , Vtable
            >
        > : mpl::true_
        {};

        namespace ext {
            template <
                typename Sig, typename IArchive, typename OArchive
              , typename Vtable
            >
            struct guid_impl<
                ::hpx::util::detail::vtable_ptr<
                    Sig
                  , IArchive
                  , OArchive
                  , Vtable
                >
            >
            {
                static inline const char * call()
                {
                    return
                        hpx::util::detail::type_hash<
                            ::hpx::util::detail::vtable_ptr<
                                Sig
                              , IArchive
                              , OArchive
                              , Vtable
                            >
                        >();
                }
            };
        }
    }
    namespace archive { namespace detail { namespace extra_detail {
        template <
            typename Sig, typename IArchive, typename OArchive
          , typename Vtable
        >
        struct init_guid<
            ::hpx::util::detail::vtable_ptr<
                Sig
              , IArchive
              , OArchive
              , Vtable
            >
        >
        {
            static
                hpx::util::detail::guid_initializer_helper<
                    ::hpx::util::detail::vtable_ptr<
                        Sig
                      , IArchive
                      , OArchive
                      , Vtable
                    >
                > const &
                g;
        };

        template <
            typename Sig, typename IArchive, typename OArchive
          , typename Vtable
        >
        hpx::util::detail::guid_initializer_helper<
            ::hpx::util::detail::vtable_ptr<
                Sig
              , IArchive
              , OArchive
              , Vtable
            >
        > const &
        init_guid<
            ::hpx::util::detail::vtable_ptr<
                Sig
              , IArchive
              , OArchive
              , Vtable
            >
        >::g = ::boost::serialization::singleton<
            hpx::util::detail::guid_initializer_helper<
                ::hpx::util::detail::vtable_ptr<
                    Sig
                  , IArchive
                  , OArchive
                  , Vtable
                >
            >
        >::get_mutable_instance().export_guid();
    }}}
}
// boost serialization registration - end

#endif

