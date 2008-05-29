//Copyright (c) 2006-2008 Emil Dotchevski and Reverge Studios, Inc.

//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef UUID_8D22C4CA9CC811DCAA9133D256D89593
#define UUID_8D22C4CA9CC811DCAA9133D256D89593

#include <boost/type.hpp>
#include <boost/exception/exception.hpp>
#include <boost/exception/error_info.hpp>
#include <boost/exception/to_string_stub.hpp>
#include <boost/current_function.hpp>
#include <boost/shared_ptr.hpp>
#include <map>

#define BOOST_ERROR_INFO\
    ::boost::throw_function(BOOST_CURRENT_FUNCTION) <<\
    ::boost::throw_file(__FILE__) <<\
    ::boost::throw_line((int)__LINE__)

namespace
boost
    {
    typedef error_info<struct tag_throw_function,char const *> throw_function;
    typedef error_info<struct tag_throw_file,char const *> throw_file;
    typedef error_info<struct tag_throw_line,int> throw_line;

    namespace
    exception_detail
        {
        class
        error_info_base
            {
            public:

            virtual std::type_info const & tag_typeid() const = 0;
            virtual std::string value_as_string() const = 0;

            protected:

#if BOOST_WORKAROUND( __GNUC__, BOOST_TESTED_AT(4) )
virtual //Disable bogus GCC warning.
#endif
            ~error_info_base()
                {
                }
            };
        }

    template <class Tag,class T>
    class
    error_info:
        public exception_detail::error_info_base
        {
        public:

        typedef T value_type;

        error_info( value_type const & value ):
            value_(value)
            {
            }

        value_type const &
        value() const
            {
            return value_;
            }

        private:

        std::type_info const &
        tag_typeid() const
            {
            return typeid(type<Tag>);
            }

        std::string
        value_as_string() const
            {
            return to_string_stub(value_);
            }

        value_type const value_;
        };

    template <class ErrorInfo>
    struct
    error_info_type
        {
        typedef typename ErrorInfo::value_type value_type;
        };

    template <class E,class Tag,class T>
    E const &
    operator<<( E const & x, error_info<Tag,T> const & v )
        {
        shared_ptr< error_info<Tag,T> > p( new error_info<Tag,T>(v) );
        x.set(p);
        return x;
        }

    template <class ErrorInfo,class E>
    shared_ptr<typename ErrorInfo::value_type const>
    get_error_info( E const & some_exception )
        {
        if( exception const * x = dynamic_cast<exception const *>(&some_exception) )
            if( shared_ptr<exception_detail::error_info_base const> eib = x->get(typeid(ErrorInfo)) )
                {
                BOOST_ASSERT( 0!=dynamic_cast<ErrorInfo const *>(eib.get()) );
                ErrorInfo const * w = static_cast<ErrorInfo const *>(eib.get());
                return shared_ptr<typename ErrorInfo::value_type const>(eib,&w->value());
                }
        return shared_ptr<typename ErrorInfo::value_type const>();
        }

    namespace
    exception_detail
        {
        class
        error_info_container_impl:
            public error_info_container
            {
            public:

            error_info_container_impl():
                count_(0)
                {
                }

            ~error_info_container_impl() throw()
                {
                }

            shared_ptr<error_info_base const>
            get( std::type_info const & ti ) const
                {
                error_info_map::const_iterator i=info_.find(typeinfo(ti));
                if( info_.end()!=i )
                    {
                    shared_ptr<error_info_base const> const & p = i->second;
                    BOOST_ASSERT( typeid(*p)==ti );
                    return p;
                    }
                return shared_ptr<error_info_base const>();
                }

            void
            set( shared_ptr<error_info_base const> const & x )
                {
                BOOST_ASSERT(x);
                info_[typeinfo(typeid(*x))] = x;
                what_.clear();
                }

            char const *
            what( std::type_info const & exception_type ) const
                {
                if( what_.empty() )
                    {
                    std::string tmp(exception_type.name());
                    tmp += '\n';
                    for( error_info_map::const_iterator i=info_.begin(),end=info_.end(); i!=end; ++i )
                        {
                        shared_ptr<error_info_base const> const & x = i->second;
                        tmp += '[';
                        tmp += x->tag_typeid().name();
                        tmp += "] = ";
                        tmp += x->value_as_string();
                        tmp += '\n';
                        }
                    what_.swap(tmp);
                    }
                return what_.c_str();
                }

            private:

            friend class exception;

            struct
            typeinfo
                {
                std::type_info const * type;

                explicit
                typeinfo( std::type_info const & t ):
                    type(&t)
                    {
                    }

                bool
                operator<( typeinfo const & b ) const
                    {
                    return 0!=(type->before(*b.type));
                    }
                };

            typedef std::map< typeinfo, shared_ptr<error_info_base const> > error_info_map;
            error_info_map info_;
            std::string mutable what_;
            int mutable count_;

            void
            add_ref() const
                {
                ++count_;
                }

            void
            release() const
                {
                if( !--count_ )
                    delete this;
                }
            };
        }

    inline
    void
    exception::
    set( shared_ptr<exception_detail::error_info_base const> const & x ) const
        {
        if( !data_ )
            data_ = intrusive_ptr<exception_detail::error_info_container>(new exception_detail::error_info_container_impl);
        data_->set(x);
        }

    inline
    shared_ptr<exception_detail::error_info_base const>
    exception::
    get( std::type_info const & ti ) const
        {
        if( data_ )
            return data_->get(ti);
        else
            return shared_ptr<exception_detail::error_info_base const>();
        }
    }

#endif
