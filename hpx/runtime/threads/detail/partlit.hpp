/*=============================================================================
    Copyright (c) 2001-2011 Joel de Guzman
    Copyright (c) 2001-2013 Hartmut Kaiser
    Copyright (c)      2010 Bryce Lelbach

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/

// make inspect happy: hpxinspect:nodeprecatedinclude hpxinspect:nodeprecatedname

#if !defined(HPX_RUNTIME_THREADS_DETAIL_PARTLIT_FEB_02_2013_0845PM)
#define HPX_RUNTIME_THREADS_DETAIL_PARTLIT_FEB_02_2013_0845PM

#if defined(HPX_MSVC)
#pragma once
#endif

#include <boost/detail/workaround.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/mpl/if.hpp>
#include <boost/spirit/home/qi/auxiliary/lazy.hpp>
#include <boost/spirit/home/qi/detail/enable_lit.hpp>
#include <boost/spirit/home/qi/detail/string_parse.hpp>
#include <boost/spirit/home/qi/domain.hpp>
#include <boost/spirit/home/qi/meta_compiler.hpp>
#include <boost/spirit/home/qi/parser.hpp>
#include <boost/spirit/home/qi/skip_over.hpp>
#include <boost/spirit/home/support/char_class.hpp>
#include <boost/spirit/home/support/common_terminals.hpp>
#include <boost/spirit/home/support/detail/get_encoding.hpp>
#include <boost/spirit/home/support/handles_container.hpp>
#include <boost/spirit/home/support/info.hpp>
#include <boost/spirit/home/support/modify.hpp>
#include <boost/spirit/home/support/string_traits.hpp>
#include <boost/spirit/home/support/unused.hpp>
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/add_reference.hpp>
#include <boost/utility/enable_if.hpp>
#include <string>

namespace hpx { namespace threads { namespace detail
{
    template <typename Char, typename Iterator, typename Attribute>
    inline bool partial_string_parse(
        Char const* str
      , Iterator& first, Iterator const& last, Attribute& attr)
    {
        Iterator i = first;
        Char ch = *str;

        for (; !!ch; ++i)
        {
            if (i == last || (ch != *i))
            {
                if (i == first)
                    return false;
                break;
            }
            ch = *++str;
        }

        boost::spirit::traits::assign_to(first, i, attr);
        first = i;
        return true;
    }

    template <typename String, typename Iterator, typename Attribute>
    inline bool partial_string_parse(
        String const& str
      , Iterator& first, Iterator const& last, Attribute& attr)
    {
        Iterator i = first;
        typename String::const_iterator stri = str.begin();
        typename String::const_iterator str_last = str.end();

        for (; stri != str_last; ++stri, ++i)
        {
            if (i == last || (*stri != *i))
            {
                if (i == first)
                    return false;
                break;
            }
        }

        boost::spirit::traits::assign_to(first, i, attr);
        first = i;
        return true;
    }

    template <typename Char, typename Iterator, typename Attribute>
    inline bool partial_string_parse(
        Char const* uc_i, Char const* lc_i
      , Iterator& first, Iterator const& last, Attribute& attr)
    {
        Iterator i = first;

        for (; *uc_i && *lc_i; ++uc_i, ++lc_i, ++i)
        {
            if (i == last || ((*uc_i != *i) && (*lc_i != *i)))
            {
                if (i == first)
                    return false;
                break;
            }
        }

        boost::spirit::traits::assign_to(first, i, attr);
        first = i;
        return true;
    }

    template <typename String, typename Iterator, typename Attribute>
    inline bool partial_string_parse(
        String const& ucstr, String const& lcstr
      , Iterator& first, Iterator const& last, Attribute& attr)
    {
        typename String::const_iterator uc_i = ucstr.begin();
        typename String::const_iterator uc_last = ucstr.end();
        typename String::const_iterator lc_i = lcstr.begin();
        Iterator i = first;

        for (; uc_i != uc_last; ++uc_i, ++lc_i, ++i)
        {
            if (i == last || ((*uc_i != *i) && (*lc_i != *i)))
            {
                if (i == first)
                    return false;
                break;
            }
        }

        boost::spirit::traits::assign_to(first, i, attr);
        first = i;
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    BOOST_SPIRIT_TERMINAL_NAME_EX(partlit, partlit_type)
}}}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace spirit
{
    // enables partlit(...)
    template <typename A0>
    struct use_terminal<qi::domain
          , terminal_ex<hpx::threads::detail::tag::partlit, fusion::vector1<A0> >
          , typename enable_if<traits::is_string<A0> >::type>
      : mpl::true_ {};
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // Parse for (possibly partial) literal strings
    template <typename String, bool no_attribute>
    struct partial_literal_string //-V690
      : boost::spirit::qi::primitive_parser<
            partial_literal_string<String, no_attribute> >
    {
        typedef typename boost::remove_const<
            typename boost::spirit::traits::char_type_of<String>::type
        >::type char_type;
        typedef std::basic_string<char_type> string_type;

        partial_literal_string(typename boost::add_reference<String>::type str_)
          : str(str_)
        {}

        template <typename Context, typename Iterator>
        struct attribute
        {
            typedef typename boost::mpl::if_c<
                no_attribute, boost::spirit::unused_type, string_type>::type
            type;
        };

        template <typename Iterator, typename Context
          , typename Skipper, typename Attribute>
        bool parse(Iterator& first, Iterator const& last
          , Context& /*context*/, Skipper const& skipper, Attribute& attr_) const
        {
            boost::spirit::qi::skip_over(first, last, skipper);
            return partial_string_parse(str, first, last, attr_);
        }

        template <typename Context>
        boost::spirit::info what(Context& /*context*/) const
        {
            return boost::spirit::info("partial-literal-string", str);
        }

        String str;

    private:
        // silence MSVC warning C4512: assignment operator could not be generated
        partial_literal_string& operator= (partial_literal_string const&);
    };

    template <typename String, bool no_attribute>
    struct no_case_partial_literal_string
      : boost::spirit::qi::primitive_parser<
            no_case_partial_literal_string<String, no_attribute> >
    {
        typedef typename boost::remove_const<
            typename boost::spirit::traits::char_type_of<String>::type
        >::type char_type;
        typedef std::basic_string<char_type> string_type;

        template <typename CharEncoding>
        no_case_partial_literal_string(char_type const* in, CharEncoding encoding)
          : str_lo(in)
          , str_hi(in)
        {
            typename string_type::iterator loi = str_lo.begin();
            typename string_type::iterator hii = str_hi.begin();

            for (; loi != str_lo.end(); ++loi, ++hii, ++in)
            {
                typedef typename CharEncoding::char_type encoded_char_type;

                *loi = static_cast<char_type>(encoding.tolower(encoded_char_type(*loi)));
                *hii = static_cast<char_type>(encoding.toupper(encoded_char_type(*hii)));
            }
        }

        template <typename Context, typename Iterator>
        struct attribute
        {
            typedef typename boost::mpl::if_c<
                no_attribute, boost::spirit::unused_type, string_type>::type
            type;
        };

        template <typename Iterator, typename Context
          , typename Skipper, typename Attribute>
        bool parse(Iterator& first, Iterator const& last
          , Context& /*context*/, Skipper const& skipper, Attribute& attr_) const
        {
            boost::spirit::qi::skip_over(first, last, skipper);
            return partial_string_parse(str_lo, str_hi, first, last, attr_);
        }

        template <typename Context>
        boost::spirit::info what(Context& /*context*/) const
        {
            return boost::spirit::info("no-case-partial-literal-string", str_lo);
        }

        string_type str_lo, str_hi;
    };
}}}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace spirit { namespace qi
{
    // Parser generators: make_xxx function (objects)

    // partlit("...")
    template <typename Modifiers, typename A0>
    struct make_primitive<
        terminal_ex<hpx::threads::detail::tag::partlit, fusion::vector1<A0> >
      , Modifiers
      , typename enable_if<traits::is_string<A0> >::type>
    {
        typedef has_modifier<Modifiers, tag::char_code_base<tag::no_case> > no_case;

        typedef typename add_const<A0>::type const_string;
        typedef typename mpl::if_<
            no_case
          , hpx::threads::detail::no_case_partial_literal_string<const_string, true>
          , hpx::threads::detail::partial_literal_string<const_string, true>
        >::type result_type;

        template <typename Terminal>
        result_type operator()(Terminal const& term, unused_type) const
        {
            return op(fusion::at_c<0>(term.args), no_case());
        }

        template <typename String>
        result_type op(String const& str, mpl::false_) const
        {
            return result_type(str);
        }

        template <typename String>
        result_type op(String const& str, mpl::true_) const
        {
            typedef typename traits::char_encoding_from_char<
                typename traits::char_type_of<A0>::type>::type encoding_type;
            typename spirit::detail::get_encoding<Modifiers,
                encoding_type>::type encoding;
            return result_type(traits::get_c_string(str), encoding);
        }
    };
}}}

namespace boost { namespace spirit { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename String, bool no_attribute, typename Attribute
      ,typename Context, typename Iterator>
    struct handles_container<
            hpx::threads::detail::partial_literal_string<String, no_attribute>
      , Attribute, Context, Iterator>
      : mpl::true_ {};

    template <typename String, bool no_attribute, typename Attribute
      , typename Context, typename Iterator>
    struct handles_container<
            hpx::threads::detail::no_case_partial_literal_string<String, no_attribute>
      , Attribute, Context, Iterator>
      : mpl::true_ {};
}}}

#endif
