//  Copyright (c) 2005-2007 Andre Merzky
//  Copyright (c) 2005-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// System Header Files
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdarg>

#include <list>
#include <vector>
#include <iostream>
#include <fstream>

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/assert.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/bind.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/assign/std/vector.hpp>

#ifdef __APPLE__
#include <crt_externs.h>
#define environ (*_NSGetEnviron())
#elif !defined(BOOST_WINDOWS)
extern char **environ;
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{

///////////////////////////////////////////////////////////////////////////////
// example ini line: line # comment
const char pattern_comment[] =  "^([^#]*)(#.*)$";

// example uses ini line: [sec.ssec]
const char pattern_section[] = "^\\[([^\\]]+)\\]$";

// example uses ini line: sec.ssec.key = val
const char pattern_qualified_entry[] = "^([^\\s=]+)\\.([^\\s=\\.]+)\\s*=\\s*(.*[^\\s]?)\\s*$";

// example uses ini line: key = val
const char pattern_entry[] = "^([^\\s=]+)\\s*=\\s*(.*[^\\s]?)\\s*$";

///////////////////////////////////////////////////////////////////////////////

namespace
{
    ///////////////////////////////////////////////////////////////////////////
    inline std::string
    trim_whitespace (std::string const &s)
    {
        typedef std::string::size_type size_type;

        size_type first = s.find_first_not_of (" \t\r\n");
        if (std::string::npos == first)
            return (std::string ());

        size_type last = s.find_last_not_of (" \t\r\n");
        return s.substr (first, last - first + 1);
    }

    ///////////////////////////////////////////////////////////////////////////
    inline section*
    add_section_if_new(section* current, std::string const& sec_name)
    {
        // do we know this one?
        if (!current->has_section(sec_name)) {
            // no - add it!
            section sec;
            current->add_section (sec_name, sec, current->get_root());
        }
        return current->get_section (sec_name);
    }

} // namespace

///////////////////////////////////////////////////////////////////////////////
section::section ()
  : root_(this_())
{
    regex_init();
}

section::section (std::string const& filename, section* root)
  : root_(NULL != root ? root : this_()), name_(filename)
{
    read(filename);
}

section::section (const section & in)
  : root_(this_()), name_(in.get_name()), parent_name_(in.get_parent_name())
{
    regex_init();

    entry_map const& e = in.get_entries();
    entry_map::const_iterator end = e.end();
    for (entry_map::const_iterator i = e.begin (); i != end; ++i)
        add_entry(i->first, i->second);

    section_map s = in.get_sections();
    section_map::iterator send = s.end();
    for (section_map::iterator si = s.begin(); si != send; ++si)
        add_section(si->first, si->second, get_root());
}

section& section::operator=(section const& rhs)
{
    if (this != &rhs) {
        root_ = this;
        parent_name_ = rhs.get_parent_name();
        name_ = rhs.get_name();

        entry_map const& e = rhs.get_entries();
        entry_map::const_iterator end = e.end();
        for (entry_map::const_iterator i = e.begin (); i != end; ++i)
            add_entry(i->first, i->second);

        section_map s = rhs.get_sections();
        section_map::iterator send = s.end();
        for (section_map::iterator si = s.begin(); si != send; ++si)
            add_section(si->first, si->second, get_root());
    }
    return *this;
}

section& section::clone_from(section const& rhs, section* root)
{
    if (this != &rhs) {
        root_ = root ? root : this;
        parent_name_ = rhs.get_parent_name();
        name_ = rhs.get_name();

        entry_map const& e = rhs.get_entries();
        entry_map::const_iterator end = e.end();
        for (entry_map::const_iterator i = e.begin (); i != end; ++i)
            add_entry(i->first, i->second);

        section_map s = rhs.get_sections();
        section_map::iterator send = s.end();
        for (section_map::iterator si = s.begin(); si != send; ++si)
            add_section(si->first, si->second, get_root());
    }
    return *this;
}

void section::read (std::string const& filename)
{
#if defined(__AIX__) && defined(__GNUC__)
    // NEVER ask why... seems to be some weird stdlib initialization problem
    // If you don't call getline() here the while(getline...) loop below will
    // crash with a bad_cast exception. Stupid AIX...
    std::string l1;
    std::ifstream i1;
    i1.open(filename.c_str(), std::ios::in);
    std::getline(i1, l1);
    i1.close();
#endif

    // build ini - open file and parse each line
    std::ifstream input(filename.c_str (), std::ios::in);
    if (!input.is_open())
        line_msg("Cannot open file: ", filename);

    // initialize
    if (!regex_init())
        line_msg ("Cannot init regex for ", filename);

    // read file
    std::string line;
    std::vector <std::string> lines;
    while (std::getline(input, line))
        lines.push_back (line);

    // parse file
    parse(filename, lines, false);
}

bool force_entry(std::string& str)
{
    std::string::size_type p = str.find_last_of("!");
    if (p != std::string::npos && str.find_first_not_of(" \t", p+1) == std::string::npos)
    {
        str = str.substr(0, p);   // remove forcing modifier ('!')
        return true;
    }
    return false;
}

// parse file
void section::parse (std::string const& sourcename,
    std::vector<std::string> const& lines, bool verify_existing)
{
    int linenum = 0;
    section* current = this;

    boost::regex regex_comment (pattern_comment, boost::regex::perl | boost::regex::icase);
    boost::regex regex_section (pattern_section, boost::regex::perl | boost::regex::icase);
    boost::regex regex_qualified_entry
        (pattern_qualified_entry, boost::regex::perl | boost::regex::icase);
    boost::regex regex_entry (pattern_entry,   boost::regex::perl | boost::regex::icase);

    std::vector<std::string>::const_iterator end = lines.end();
    for (std::vector<std::string>::const_iterator it = lines.begin();
         it != end; ++it)
    {
        ++linenum;

        // remove trailing new lines and white spaces
        std::string line(trim_whitespace (*it));

        // skip if empty line
        if (line.empty())
            continue;

        // weep out comments
        boost::smatch what_comment;
        if (boost::regex_match (line, what_comment, regex_comment))
        {
            BOOST_ASSERT(3 == what_comment.size());

            line = trim_whitespace (what_comment[1]);
            if (line.empty())
                continue;
        }

        // no comments anymore: line is either section, key=val,
        // or garbage/empty
        boost::smatch what;
        if (boost::regex_match(line, what, regex_qualified_entry))
        {
            // found a entry line
            if (4 != what.size()) //-V112
            {
                line_msg("Cannot parse key/value in: ", sourcename, linenum, line);
            }

            section* s = current;  // save the section we're in
            current = this;           // start adding sections at the root

            // got the section name. It might be hierarchical, so split it up, and
            // for each elem, check if we have it.  If not, create it, and add
            std::string sec_name (what[1]);
            std::string::size_type pos = 0;
            for (std::string::size_type pos1 = sec_name.find_first_of('.');
                 std::string::npos != pos1;
                 pos1 = sec_name.find_first_of ('.', pos = pos1 + 1))
            {
                current = add_section_if_new(current, sec_name.substr(pos, pos1 - pos));
            }

            current = add_section_if_new (current, sec_name.substr(pos));

            // add key/val to this section
            std::string key(what[2]);
            if (!force_entry(key) && verify_existing && !current->has_entry(key)) 
            {
                line_msg ("Attempt to initialize unknown entry: ", sourcename, 
                    linenum, line);
            }
            current->add_entry (key, what[3]);

            // restore the old section
            current = s;
        }

        else if (boost::regex_match(line, what, regex_section))
        {
            // found a section line
            if (2 != what.size())
            {
                line_msg("Cannot parse section in: ", sourcename, linenum, line);
            }

            current = this;     // start adding sections at the root

            // got the section name. It might be hierarchical, so split it up, and
            // for each elem, check if we have it.  If not, create it, and add
            std::string sec_name (what[1]);
            std::string::size_type pos = 0;
            for (std::string::size_type pos1 = sec_name.find_first_of('.');
                 std::string::npos != pos1;
                 pos1 = sec_name.find_first_of ('.', pos = pos1 + 1))
            {
                current = add_section_if_new(current, sec_name.substr(pos, pos1 - pos));
            }

            current = add_section_if_new (current, sec_name.substr(pos));
        }

        // did not match section, so might be key/val entry
        else if ( boost::regex_match (line, what, regex_entry) )
        {
            // found a entry line
            if (3 != what.size())
            {
                line_msg("Cannot parse key/value in: ", sourcename, linenum, line);
            }

            // add key/val to current section
            std::string key(what[1]);
            if (!force_entry(key) && verify_existing && !current->has_entry(key)) 
            {
                line_msg ("Attempt to initialize unknown entry: ", sourcename, 
                    linenum, line);
            }
            current->add_entry (key, what[2]);
        }
        else {
            // Hmm, is not a section, is not an entry, is not empty - must be 
            // an error!
            line_msg ("Cannot parse line at: ", sourcename, linenum, line);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
void section::add_section (std::string const& sec_name, section& sec, section* root)
{
    // setting name and root
    sec.name_ = sec_name;
    sec.parent_name_ = get_full_name();

    section& newsec = sections_[sec_name];
    newsec.clone_from(sec, (NULL != root) ? root : get_root());
}

bool section::has_section (std::string const& sec_name) const
{
    std::string::size_type i = sec_name.find(".");
    if (i != std::string::npos)
    {
        std::string cor_sec_name = sec_name.substr (0, i);

        section_map::const_iterator it = sections_.find(cor_sec_name);
        if (it != sections_.end())
        {
            std::string sub_sec_name = sec_name.substr(i+1);
            return (*it).second.has_section(sub_sec_name);
        }
        return false;
    }
    return sections_.find(sec_name) != sections_.end();
}

section* section::get_section (std::string const& sec_name)
{
    std::string::size_type i  = sec_name.find (".");
    if (i != std::string::npos)
    {
        std::string cor_sec_name = sec_name.substr (0, i);
        section_map::iterator it = sections_.find(cor_sec_name);
        if (it != sections_.end())
        {
            std::string sub_sec_name = sec_name.substr(i+1);
            return (*it).second.get_section(sub_sec_name);
        }

        std::string name(get_name());
        if (name.empty())
            name = "<root>";

        HPX_THROW_EXCEPTION(bad_parameter, "section::get_section",
            "No such section (" + sec_name + ") in section: " + name);
        return NULL;
    }

    section_map::iterator it = sections_.find(sec_name);
    if (it != sections_.end())
        return &((*it).second);

    HPX_THROW_EXCEPTION(bad_parameter, "section::get_section",
        "No such section (" + sec_name + ") in section: " + get_name());
    return NULL;
}

section const* section::get_section (std::string const& sec_name) const
{
    std::string::size_type i  = sec_name.find (".");
    if (i != std::string::npos)
    {
        std::string cor_sec_name = sec_name.substr (0,  i);
        section_map::const_iterator it = sections_.find(cor_sec_name);
        if (it != sections_.end())
        {
            std::string sub_sec_name = sec_name.substr(i+1);
            return (*it).second.get_section(sub_sec_name);
        }

        std::string name(get_name());
        if (name.empty())
            name = "<root>";

        HPX_THROW_EXCEPTION(bad_parameter, "section::get_section",
            "No such section (" + sec_name + ") in section: " + name);
        return NULL;
    }

    section_map::const_iterator it = sections_.find(sec_name);
    if (it != sections_.end())
        return &((*it).second);

    HPX_THROW_EXCEPTION(bad_parameter, "section::get_section",
        "No such section (" + sec_name + ") in section: " + get_name());
    return NULL;
}

void section::add_entry (std::string const& key, std::string val)
{
    // first expand the full property name in the value (avoids infinite recursion)
    this->expand_only(val, std::string::size_type(-1), get_full_name() + "." + key);

    // now add this entry to the section
    entries_[key] = val;
}

bool section::has_entry (std::string const& key) const
{
    std::string::size_type i = key.find (".");
    if (i != std::string::npos)
    {
        std::string sub_sec = key.substr(0, i);
        std::string sub_key = key.substr(i+1, key.size() - i);
        if (has_section(sub_sec))
        {
            section_map::const_iterator cit = sections_.find(sub_sec);
            BOOST_ASSERT(cit != sections_.end());
            return (*cit).second.has_entry(sub_key);
        }
        return false;
    }
    return entries_.find(key) != entries_.end();
}

std::string section::get_entry (std::string const& key) const
{
    std::string::size_type i = key.find (".");
    if (i != std::string::npos)
    {
        std::string sub_sec = key.substr(0, i);
        std::string sub_key = key.substr(i+1, key.size() - i);
        if (has_section(sub_sec))
        {
            section_map::const_iterator cit = sections_.find(sub_sec);
            BOOST_ASSERT(cit != sections_.end());
            return (*cit).second.get_entry(sub_key);
        }

        HPX_THROW_EXCEPTION(bad_parameter, "section::get_entry",
            "No such key (" + key + ") in section: " + get_name());
        return "";
    }

    if (entries_.find(key) != entries_.end())
    {
        entry_map::const_iterator cit = entries_.find(key);
        BOOST_ASSERT(cit != entries_.end());
        return this->expand((*cit).second);
    }

    HPX_THROW_EXCEPTION(bad_parameter, "section::get_entry",
        "No such section (" + key + ") in section: " + get_name());
    return "";
}

std::string
section::get_entry (std::string const& key, std::string const& default_val) const
{
    typedef std::vector<std::string> string_vector;

    string_vector split_key;
    boost::split(split_key, key, boost::is_any_of("."));

    std::string sk = split_key.back();
    split_key.pop_back();

    section const* cur_section = this;
    for (string_vector::const_iterator iter = split_key.begin(),
         end = split_key.end(); iter != end; ++iter)
    {
        section_map::const_iterator next = cur_section->sections_.find(*iter);
        if (cur_section->sections_.end() == next)
            return this->expand(default_val);
        cur_section = &next->second;
    }

    entry_map::const_iterator entry = cur_section->entries_.find(sk);
    if (cur_section->entries_.end() == entry)
        return this->expand(default_val);

    return this->expand(entry->second);
}

inline void indent (int ind, std::ostream& strm)
{
    for (int i = 0; i < ind; ++i)
        strm << "  ";
}

void section::dump(int ind, std::ostream& strm) const
{
    bool header = false;
    if (0 == ind)
        header = true;

    ++ind;
    if (header)
    {
        if(get_root() == this) {
            strm << "============================\n";
        }
        else {
            strm << "============================[\n"
                 << get_name() << "\n" << "]\n";
        }
    }

    entry_map::const_iterator eend = entries_.end();
    for (entry_map::const_iterator i = entries_.begin(); i != eend; ++i)
    {
        indent (ind, strm);

        const std::string expansion = this->expand(i->second);

        // Check if the expanded entry is different from the actual entry.
        if (expansion != i->second)
            // If the expansion is different from the real entry, then print
            // it out.
            strm << "'" << i->first << "' : '" << i->second << "' -> '"
                 << expansion << "'\n";
        else
            strm << "'" << i->first << "' : '" << i->second << "'\n";
    }

    section_map::const_iterator send = sections_.end();
    for (section_map::const_iterator i  = sections_.begin(); i != send; ++i)
    {
        indent (ind, strm);
        strm << "[" << i->first << "]\n";
        (*i).second.dump (ind, strm);
    }

    if (header)
        strm << "============================\n";

    strm << std::flush;
}

void section::merge(std::string const& filename)
{
    section tmp(filename, root_);
    merge(tmp);
}

void section::merge(section& second)
{
    // merge entries: keep own entries, and add other entries
    entry_map const& s_entries = second.get_entries();
    entry_map::const_iterator end = s_entries.end();
    for (entry_map::const_iterator i = s_entries.begin(); i != end; ++i)
        entries_[i->first] = i->second;

    // merge subsection known in first section
    section_map::iterator send = sections_.end();
    for (section_map::iterator i = sections_.begin(); i != send; ++i)
    {
        // is there something to merge with?
        if (second.has_section(i->first))
            i->second.merge (second.sections_[i->first]);
    }

    // merge subsection known in second section
    section_map s = second.get_sections();
    section_map::iterator secend = s.end();
    for (section_map::iterator i = s.begin (); i != secend; ++i)
    {
        // if THIS knows the section, we already merged it above
        if (!has_section(i->first))
        {
            // it is not known here, so we can't merge, but have to add it.
            add_section (i->first, i->second, get_root());
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////
bool section::regex_init (void)
{
    return true;
}

void section::line_msg(std::string msg, std::string const& file,
    int lnum, std::string const& line)
{
    msg += " " + file;
    if (lnum > 0)
        msg += ": line " + boost::lexical_cast<std::string>(lnum);
    if (!line.empty())
        msg += " (offending entry: " + line + ")";

    HPX_THROW_EXCEPTION(no_success, "section::line_msg", msg);
}

///////////////////////////////////////////////////////////////////////////////
// find the matching closing brace starting from 'begin', escaped braces will
// be un-escaped
inline std::string::size_type
find_next(char const* ch, std::string& value,
    std::string::size_type begin = static_cast<std::string::size_type>(-1))
{
    std::string::size_type end = value.find_first_of(ch, begin+1);
    while (end != std::string::npos) {
        if (end != 0 && value[end-1] != '\\')
            break;
        value.replace(end-1, 2, ch);
        end = value.find_first_of(ch, end);
    }
    return end;
}

///////////////////////////////////////////////////////////////////////////////
void section::expand(std::string& value, std::string::size_type begin) const
{
    std::string::size_type p = value.find_first_of("$", begin+1);
    while (p != std::string::npos && value.size()-1 != p) {
        if ('[' == value[p+1])
            expand_bracket(value, p);
        else if ('{' == value[p+1])
            expand_brace(value, p);
        p = value.find_first_of("$", p+1);
    }
}

void section::expand_bracket(std::string& value, std::string::size_type begin) const
{
    // expand all keys embedded inside this key
    this->expand(value, begin);

    // now expand the key itself
    std::string::size_type end = find_next("]", value, begin+1);
    if (end != std::string::npos)
    {
        std::string to_expand = value.substr(begin+2, end-begin-2);
        std::string::size_type colon = find_next(":", to_expand);
        if (colon == std::string::npos) {
            value.replace(begin, end-begin+1, root_->get_entry(to_expand, std::string("")));
        }
        else {
            value.replace(begin, end-begin+1,
                root_->get_entry(to_expand.substr(0, colon), to_expand.substr(colon+1)));
        }
    }
}

void section::expand_brace(std::string& value, std::string::size_type begin) const
{
    // expand all keys embedded inside this key
    this->expand(value, begin);

    // now expand the key itself
    std::string::size_type end = find_next("}", value, begin+1);
    if (end != std::string::npos)
    {
        std::string to_expand = value.substr(begin+2, end-begin-2);
        std::string::size_type colon = find_next(":", to_expand);
        if (colon == std::string::npos) {
            char* env = getenv(to_expand.c_str());
            value.replace(begin, end-begin+1, 0 != env ? env : "");
        }
        else {
            char* env = getenv(to_expand.substr(0, colon).c_str());
            value.replace(begin, end-begin+1,
                0 != env ? std::string(env) : to_expand.substr(colon+1));
        }
    }
}

std::string section::expand (std::string value) const
{
    expand(value, std::string::size_type(-1));
    return value;
}

///////////////////////////////////////////////////////////////////////////////
void section::expand_only(std::string& value, std::string::size_type begin,
    std::string const& expand_this) const
{
    std::string::size_type p = value.find_first_of("$", begin+1);
    while (p != std::string::npos && value.size()-1 != p) {
        if ('[' == value[p+1])
            expand_bracket_only(value, p, expand_this);
        else if ('{' == value[p+1])
            expand_brace_only(value, p, expand_this);
        p = value.find_first_of("$", p+1);
    }
}

void section::expand_bracket_only(std::string& value,
    std::string::size_type begin, std::string const& expand_this) const
{
    // expand all keys embedded inside this key
    this->expand_only(value, begin, expand_this);

    // now expand the key itself
    std::string::size_type end = find_next("]", value, begin+1);
    if (end != std::string::npos)
    {
        std::string to_expand = value.substr(begin+2, end-begin-2);
        std::string::size_type colon = find_next(":", to_expand);
        if (colon == std::string::npos) {
            if (to_expand == expand_this) {
                value.replace(begin, end-begin+1,
                    root_->get_entry(to_expand, std::string("")));
            }
        }
        else if (to_expand.substr(0, colon) == expand_this) {
            value.replace(begin, end-begin+1,
                root_->get_entry(to_expand.substr(0, colon),
                to_expand.substr(colon+1)));
        }
    }
}

void section::expand_brace_only(std::string& value,
    std::string::size_type begin, std::string const& expand_this) const
{
    // expand all keys embedded inside this key
    this->expand_only(value, begin, expand_this);

    // now expand the key itself
    std::string::size_type end = find_next("}", value, begin+1);
    if (end != std::string::npos)
    {
        std::string to_expand = value.substr(begin+2, end-begin-2);
        std::string::size_type colon = find_next(":", to_expand);
        if (colon == std::string::npos) {
            char* env = getenv(to_expand.c_str());
            value.replace(begin, end-begin+1, 0 != env ? env : "");
        }
        else {
            char* env = getenv(to_expand.substr(0, colon).c_str());
            value.replace(begin, end-begin+1,
                0 != env ? std::string(env) : to_expand.substr(colon+1));
        }
    }
}

std::string section::expand_only(std::string value,
    std::string const& expand_this) const
{
    expand_only(value, std::string::size_type(-1), expand_this);
    return value;
}

///////////////////////////////////////////////////////////////////////////////
template <typename Archive>
void section::save(Archive& ar, const unsigned int version) const
{
    using namespace boost::serialization;

    ar << make_nvp("name", name_);
    ar << make_nvp("parent_name", parent_name_);
    ar << make_nvp("entries", entries_);
    ar << make_nvp("sections", sections_);
}

template <typename Archive>
void section::load(Archive& ar, const unsigned int version)
{
    using namespace boost::serialization;

    ar >> make_nvp("name", name_);
    ar >> make_nvp("parent_name", parent_name_);
    ar >> make_nvp("entries", entries_);
    ar >> make_nvp("sections", sections_);

    set_root(this, true);     // make this the current root
}

///////////////////////////////////////////////////////////////////////////////
// explicit instantiation for the correct archive types
template HPX_EXPORT void
section::save(util::portable_binary_oarchive&, const unsigned int version) const;

template HPX_EXPORT void
section::load(util::portable_binary_iarchive&, const unsigned int version);

}}  // namespace hpx::util

