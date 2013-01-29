// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace boost { namespace detail
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5>
    inline unsigned lock_helper(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5)
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5>
    inline unsigned try_lock_internal(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5)
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}
namespace boost
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5>
    inline void lock(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5)
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                case 0: lock_first = detail::lock_helper( m0 , m1 , m2 , m3 , m4 , m5); if (!lock_first) return; break; case 1: lock_first = detail::lock_helper( m1 , m2 , m3 , m4 , m5 , m0); if (!lock_first) return; lock_first = (lock_first+ 1) % 6; break; case 2: lock_first = detail::lock_helper( m2 , m3 , m4 , m5 , m0 , m1); if (!lock_first) return; lock_first = (lock_first+ 2) % 6; break; case 3: lock_first = detail::lock_helper( m3 , m4 , m5 , m0 , m1 , m2); if (!lock_first) return; lock_first = (lock_first+ 3) % 6; break; case 4: lock_first = detail::lock_helper( m4 , m5 , m0 , m1 , m2 , m3); if (!lock_first) return; lock_first = (lock_first+ 4) % 6; break; case 5: lock_first = detail::lock_helper( m5 , m0 , m1 , m2 , m3 , m4); if (!lock_first) return; lock_first = (lock_first+ 5) % 6; break;
            }
        }
    }
}
namespace boost { namespace detail
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6>
    inline unsigned lock_helper(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6)
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6>
    inline unsigned try_lock_internal(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6)
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}
namespace boost
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6>
    inline void lock(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6)
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                case 0: lock_first = detail::lock_helper( m0 , m1 , m2 , m3 , m4 , m5 , m6); if (!lock_first) return; break; case 1: lock_first = detail::lock_helper( m1 , m2 , m3 , m4 , m5 , m6 , m0); if (!lock_first) return; lock_first = (lock_first+ 1) % 7; break; case 2: lock_first = detail::lock_helper( m2 , m3 , m4 , m5 , m6 , m0 , m1); if (!lock_first) return; lock_first = (lock_first+ 2) % 7; break; case 3: lock_first = detail::lock_helper( m3 , m4 , m5 , m6 , m0 , m1 , m2); if (!lock_first) return; lock_first = (lock_first+ 3) % 7; break; case 4: lock_first = detail::lock_helper( m4 , m5 , m6 , m0 , m1 , m2 , m3); if (!lock_first) return; lock_first = (lock_first+ 4) % 7; break; case 5: lock_first = detail::lock_helper( m5 , m6 , m0 , m1 , m2 , m3 , m4); if (!lock_first) return; lock_first = (lock_first+ 5) % 7; break; case 6: lock_first = detail::lock_helper( m6 , m0 , m1 , m2 , m3 , m4 , m5); if (!lock_first) return; lock_first = (lock_first+ 6) % 7; break;
            }
        }
    }
}
namespace boost { namespace detail
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7>
    inline unsigned lock_helper(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7)
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7>
    inline unsigned try_lock_internal(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7)
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}
namespace boost
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7>
    inline void lock(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7)
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                case 0: lock_first = detail::lock_helper( m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7); if (!lock_first) return; break; case 1: lock_first = detail::lock_helper( m1 , m2 , m3 , m4 , m5 , m6 , m7 , m0); if (!lock_first) return; lock_first = (lock_first+ 1) % 8; break; case 2: lock_first = detail::lock_helper( m2 , m3 , m4 , m5 , m6 , m7 , m0 , m1); if (!lock_first) return; lock_first = (lock_first+ 2) % 8; break; case 3: lock_first = detail::lock_helper( m3 , m4 , m5 , m6 , m7 , m0 , m1 , m2); if (!lock_first) return; lock_first = (lock_first+ 3) % 8; break; case 4: lock_first = detail::lock_helper( m4 , m5 , m6 , m7 , m0 , m1 , m2 , m3); if (!lock_first) return; lock_first = (lock_first+ 4) % 8; break; case 5: lock_first = detail::lock_helper( m5 , m6 , m7 , m0 , m1 , m2 , m3 , m4); if (!lock_first) return; lock_first = (lock_first+ 5) % 8; break; case 6: lock_first = detail::lock_helper( m6 , m7 , m0 , m1 , m2 , m3 , m4 , m5); if (!lock_first) return; lock_first = (lock_first+ 6) % 8; break; case 7: lock_first = detail::lock_helper( m7 , m0 , m1 , m2 , m3 , m4 , m5 , m6); if (!lock_first) return; lock_first = (lock_first+ 7) % 8; break;
            }
        }
    }
}
namespace boost { namespace detail
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8>
    inline unsigned lock_helper(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8)
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8>
    inline unsigned try_lock_internal(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8)
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}
namespace boost
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8>
    inline void lock(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8)
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                case 0: lock_first = detail::lock_helper( m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8); if (!lock_first) return; break; case 1: lock_first = detail::lock_helper( m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m0); if (!lock_first) return; lock_first = (lock_first+ 1) % 9; break; case 2: lock_first = detail::lock_helper( m2 , m3 , m4 , m5 , m6 , m7 , m8 , m0 , m1); if (!lock_first) return; lock_first = (lock_first+ 2) % 9; break; case 3: lock_first = detail::lock_helper( m3 , m4 , m5 , m6 , m7 , m8 , m0 , m1 , m2); if (!lock_first) return; lock_first = (lock_first+ 3) % 9; break; case 4: lock_first = detail::lock_helper( m4 , m5 , m6 , m7 , m8 , m0 , m1 , m2 , m3); if (!lock_first) return; lock_first = (lock_first+ 4) % 9; break; case 5: lock_first = detail::lock_helper( m5 , m6 , m7 , m8 , m0 , m1 , m2 , m3 , m4); if (!lock_first) return; lock_first = (lock_first+ 5) % 9; break; case 6: lock_first = detail::lock_helper( m6 , m7 , m8 , m0 , m1 , m2 , m3 , m4 , m5); if (!lock_first) return; lock_first = (lock_first+ 6) % 9; break; case 7: lock_first = detail::lock_helper( m7 , m8 , m0 , m1 , m2 , m3 , m4 , m5 , m6); if (!lock_first) return; lock_first = (lock_first+ 7) % 9; break; case 8: lock_first = detail::lock_helper( m8 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7); if (!lock_first) return; lock_first = (lock_first+ 8) % 9; break;
            }
        }
    }
}
namespace boost { namespace detail
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9>
    inline unsigned lock_helper(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9)
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9>
    inline unsigned try_lock_internal(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9)
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}
namespace boost
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9>
    inline void lock(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9)
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                case 0: lock_first = detail::lock_helper( m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9); if (!lock_first) return; break; case 1: lock_first = detail::lock_helper( m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m0); if (!lock_first) return; lock_first = (lock_first+ 1) % 10; break; case 2: lock_first = detail::lock_helper( m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m0 , m1); if (!lock_first) return; lock_first = (lock_first+ 2) % 10; break; case 3: lock_first = detail::lock_helper( m3 , m4 , m5 , m6 , m7 , m8 , m9 , m0 , m1 , m2); if (!lock_first) return; lock_first = (lock_first+ 3) % 10; break; case 4: lock_first = detail::lock_helper( m4 , m5 , m6 , m7 , m8 , m9 , m0 , m1 , m2 , m3); if (!lock_first) return; lock_first = (lock_first+ 4) % 10; break; case 5: lock_first = detail::lock_helper( m5 , m6 , m7 , m8 , m9 , m0 , m1 , m2 , m3 , m4); if (!lock_first) return; lock_first = (lock_first+ 5) % 10; break; case 6: lock_first = detail::lock_helper( m6 , m7 , m8 , m9 , m0 , m1 , m2 , m3 , m4 , m5); if (!lock_first) return; lock_first = (lock_first+ 6) % 10; break; case 7: lock_first = detail::lock_helper( m7 , m8 , m9 , m0 , m1 , m2 , m3 , m4 , m5 , m6); if (!lock_first) return; lock_first = (lock_first+ 7) % 10; break; case 8: lock_first = detail::lock_helper( m8 , m9 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7); if (!lock_first) return; lock_first = (lock_first+ 8) % 10; break; case 9: lock_first = detail::lock_helper( m9 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8); if (!lock_first) return; lock_first = (lock_first+ 9) % 10; break;
            }
        }
    }
}
namespace boost { namespace detail
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9 , typename MutexType10>
    inline unsigned lock_helper(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9 , MutexType10 & m10)
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9 , typename MutexType10>
    inline unsigned try_lock_internal(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9 , MutexType10 & m10)
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}
namespace boost
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9 , typename MutexType10>
    inline void lock(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9 , MutexType10 & m10)
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                case 0: lock_first = detail::lock_helper( m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10); if (!lock_first) return; break; case 1: lock_first = detail::lock_helper( m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m0); if (!lock_first) return; lock_first = (lock_first+ 1) % 11; break; case 2: lock_first = detail::lock_helper( m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m0 , m1); if (!lock_first) return; lock_first = (lock_first+ 2) % 11; break; case 3: lock_first = detail::lock_helper( m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m0 , m1 , m2); if (!lock_first) return; lock_first = (lock_first+ 3) % 11; break; case 4: lock_first = detail::lock_helper( m4 , m5 , m6 , m7 , m8 , m9 , m10 , m0 , m1 , m2 , m3); if (!lock_first) return; lock_first = (lock_first+ 4) % 11; break; case 5: lock_first = detail::lock_helper( m5 , m6 , m7 , m8 , m9 , m10 , m0 , m1 , m2 , m3 , m4); if (!lock_first) return; lock_first = (lock_first+ 5) % 11; break; case 6: lock_first = detail::lock_helper( m6 , m7 , m8 , m9 , m10 , m0 , m1 , m2 , m3 , m4 , m5); if (!lock_first) return; lock_first = (lock_first+ 6) % 11; break; case 7: lock_first = detail::lock_helper( m7 , m8 , m9 , m10 , m0 , m1 , m2 , m3 , m4 , m5 , m6); if (!lock_first) return; lock_first = (lock_first+ 7) % 11; break; case 8: lock_first = detail::lock_helper( m8 , m9 , m10 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7); if (!lock_first) return; lock_first = (lock_first+ 8) % 11; break; case 9: lock_first = detail::lock_helper( m9 , m10 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8); if (!lock_first) return; lock_first = (lock_first+ 9) % 11; break; case 10: lock_first = detail::lock_helper( m10 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9); if (!lock_first) return; lock_first = (lock_first+ 10) % 11; break;
            }
        }
    }
}
namespace boost { namespace detail
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9 , typename MutexType10 , typename MutexType11>
    inline unsigned lock_helper(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9 , MutexType10 & m10 , MutexType11 & m11)
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9 , typename MutexType10 , typename MutexType11>
    inline unsigned try_lock_internal(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9 , MutexType10 & m10 , MutexType11 & m11)
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}
namespace boost
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9 , typename MutexType10 , typename MutexType11>
    inline void lock(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9 , MutexType10 & m10 , MutexType11 & m11)
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                case 0: lock_first = detail::lock_helper( m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11); if (!lock_first) return; break; case 1: lock_first = detail::lock_helper( m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m0); if (!lock_first) return; lock_first = (lock_first+ 1) % 12; break; case 2: lock_first = detail::lock_helper( m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m0 , m1); if (!lock_first) return; lock_first = (lock_first+ 2) % 12; break; case 3: lock_first = detail::lock_helper( m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m0 , m1 , m2); if (!lock_first) return; lock_first = (lock_first+ 3) % 12; break; case 4: lock_first = detail::lock_helper( m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m0 , m1 , m2 , m3); if (!lock_first) return; lock_first = (lock_first+ 4) % 12; break; case 5: lock_first = detail::lock_helper( m5 , m6 , m7 , m8 , m9 , m10 , m11 , m0 , m1 , m2 , m3 , m4); if (!lock_first) return; lock_first = (lock_first+ 5) % 12; break; case 6: lock_first = detail::lock_helper( m6 , m7 , m8 , m9 , m10 , m11 , m0 , m1 , m2 , m3 , m4 , m5); if (!lock_first) return; lock_first = (lock_first+ 6) % 12; break; case 7: lock_first = detail::lock_helper( m7 , m8 , m9 , m10 , m11 , m0 , m1 , m2 , m3 , m4 , m5 , m6); if (!lock_first) return; lock_first = (lock_first+ 7) % 12; break; case 8: lock_first = detail::lock_helper( m8 , m9 , m10 , m11 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7); if (!lock_first) return; lock_first = (lock_first+ 8) % 12; break; case 9: lock_first = detail::lock_helper( m9 , m10 , m11 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8); if (!lock_first) return; lock_first = (lock_first+ 9) % 12; break; case 10: lock_first = detail::lock_helper( m10 , m11 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9); if (!lock_first) return; lock_first = (lock_first+ 10) % 12; break; case 11: lock_first = detail::lock_helper( m11 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10); if (!lock_first) return; lock_first = (lock_first+ 11) % 12; break;
            }
        }
    }
}
namespace boost { namespace detail
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9 , typename MutexType10 , typename MutexType11 , typename MutexType12>
    inline unsigned lock_helper(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9 , MutexType10 & m10 , MutexType11 & m11 , MutexType12 & m12)
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9 , typename MutexType10 , typename MutexType11 , typename MutexType12>
    inline unsigned try_lock_internal(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9 , MutexType10 & m10 , MutexType11 & m11 , MutexType12 & m12)
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}
namespace boost
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9 , typename MutexType10 , typename MutexType11 , typename MutexType12>
    inline void lock(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9 , MutexType10 & m10 , MutexType11 & m11 , MutexType12 & m12)
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                case 0: lock_first = detail::lock_helper( m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12); if (!lock_first) return; break; case 1: lock_first = detail::lock_helper( m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m0); if (!lock_first) return; lock_first = (lock_first+ 1) % 13; break; case 2: lock_first = detail::lock_helper( m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m0 , m1); if (!lock_first) return; lock_first = (lock_first+ 2) % 13; break; case 3: lock_first = detail::lock_helper( m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m0 , m1 , m2); if (!lock_first) return; lock_first = (lock_first+ 3) % 13; break; case 4: lock_first = detail::lock_helper( m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m0 , m1 , m2 , m3); if (!lock_first) return; lock_first = (lock_first+ 4) % 13; break; case 5: lock_first = detail::lock_helper( m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m0 , m1 , m2 , m3 , m4); if (!lock_first) return; lock_first = (lock_first+ 5) % 13; break; case 6: lock_first = detail::lock_helper( m6 , m7 , m8 , m9 , m10 , m11 , m12 , m0 , m1 , m2 , m3 , m4 , m5); if (!lock_first) return; lock_first = (lock_first+ 6) % 13; break; case 7: lock_first = detail::lock_helper( m7 , m8 , m9 , m10 , m11 , m12 , m0 , m1 , m2 , m3 , m4 , m5 , m6); if (!lock_first) return; lock_first = (lock_first+ 7) % 13; break; case 8: lock_first = detail::lock_helper( m8 , m9 , m10 , m11 , m12 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7); if (!lock_first) return; lock_first = (lock_first+ 8) % 13; break; case 9: lock_first = detail::lock_helper( m9 , m10 , m11 , m12 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8); if (!lock_first) return; lock_first = (lock_first+ 9) % 13; break; case 10: lock_first = detail::lock_helper( m10 , m11 , m12 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9); if (!lock_first) return; lock_first = (lock_first+ 10) % 13; break; case 11: lock_first = detail::lock_helper( m11 , m12 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10); if (!lock_first) return; lock_first = (lock_first+ 11) % 13; break; case 12: lock_first = detail::lock_helper( m12 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11); if (!lock_first) return; lock_first = (lock_first+ 12) % 13; break;
            }
        }
    }
}
namespace boost { namespace detail
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9 , typename MutexType10 , typename MutexType11 , typename MutexType12 , typename MutexType13>
    inline unsigned lock_helper(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9 , MutexType10 & m10 , MutexType11 & m11 , MutexType12 & m12 , MutexType13 & m13)
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9 , typename MutexType10 , typename MutexType11 , typename MutexType12 , typename MutexType13>
    inline unsigned try_lock_internal(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9 , MutexType10 & m10 , MutexType11 & m11 , MutexType12 & m12 , MutexType13 & m13)
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}
namespace boost
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9 , typename MutexType10 , typename MutexType11 , typename MutexType12 , typename MutexType13>
    inline void lock(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9 , MutexType10 & m10 , MutexType11 & m11 , MutexType12 & m12 , MutexType13 & m13)
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                case 0: lock_first = detail::lock_helper( m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13); if (!lock_first) return; break; case 1: lock_first = detail::lock_helper( m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13 , m0); if (!lock_first) return; lock_first = (lock_first+ 1) % 14; break; case 2: lock_first = detail::lock_helper( m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13 , m0 , m1); if (!lock_first) return; lock_first = (lock_first+ 2) % 14; break; case 3: lock_first = detail::lock_helper( m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13 , m0 , m1 , m2); if (!lock_first) return; lock_first = (lock_first+ 3) % 14; break; case 4: lock_first = detail::lock_helper( m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13 , m0 , m1 , m2 , m3); if (!lock_first) return; lock_first = (lock_first+ 4) % 14; break; case 5: lock_first = detail::lock_helper( m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13 , m0 , m1 , m2 , m3 , m4); if (!lock_first) return; lock_first = (lock_first+ 5) % 14; break; case 6: lock_first = detail::lock_helper( m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13 , m0 , m1 , m2 , m3 , m4 , m5); if (!lock_first) return; lock_first = (lock_first+ 6) % 14; break; case 7: lock_first = detail::lock_helper( m7 , m8 , m9 , m10 , m11 , m12 , m13 , m0 , m1 , m2 , m3 , m4 , m5 , m6); if (!lock_first) return; lock_first = (lock_first+ 7) % 14; break; case 8: lock_first = detail::lock_helper( m8 , m9 , m10 , m11 , m12 , m13 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7); if (!lock_first) return; lock_first = (lock_first+ 8) % 14; break; case 9: lock_first = detail::lock_helper( m9 , m10 , m11 , m12 , m13 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8); if (!lock_first) return; lock_first = (lock_first+ 9) % 14; break; case 10: lock_first = detail::lock_helper( m10 , m11 , m12 , m13 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9); if (!lock_first) return; lock_first = (lock_first+ 10) % 14; break; case 11: lock_first = detail::lock_helper( m11 , m12 , m13 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10); if (!lock_first) return; lock_first = (lock_first+ 11) % 14; break; case 12: lock_first = detail::lock_helper( m12 , m13 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11); if (!lock_first) return; lock_first = (lock_first+ 12) % 14; break; case 13: lock_first = detail::lock_helper( m13 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12); if (!lock_first) return; lock_first = (lock_first+ 13) % 14; break;
            }
        }
    }
}
namespace boost { namespace detail
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9 , typename MutexType10 , typename MutexType11 , typename MutexType12 , typename MutexType13 , typename MutexType14>
    inline unsigned lock_helper(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9 , MutexType10 & m10 , MutexType11 & m11 , MutexType12 & m12 , MutexType13 & m13 , MutexType14 & m14)
    {
        boost::unique_lock<MutexType0> l0(m0);
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13 , m14))
        {
            return failed_lock;
        }
        l0.release();
        return 0;
    }
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9 , typename MutexType10 , typename MutexType11 , typename MutexType12 , typename MutexType13 , typename MutexType14>
    inline unsigned try_lock_internal(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9 , MutexType10 & m10 , MutexType11 & m11 , MutexType12 & m12 , MutexType13 & m13 , MutexType14 & m14)
    {
        boost::unique_lock<MutexType0> l0(m0, boost::try_to_lock);
        if (!l0)
        {
            return 1;
        }
        if (unsigned const failed_lock = try_lock_internal(m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13 , m14))
        {
            return failed_lock + 1;
        }
        l0.release();
        return 0;
    }
}}
namespace boost
{
    template <typename MutexType0 , typename MutexType1 , typename MutexType2 , typename MutexType3 , typename MutexType4 , typename MutexType5 , typename MutexType6 , typename MutexType7 , typename MutexType8 , typename MutexType9 , typename MutexType10 , typename MutexType11 , typename MutexType12 , typename MutexType13 , typename MutexType14>
    inline void lock(MutexType0 & m0 , MutexType1 & m1 , MutexType2 & m2 , MutexType3 & m3 , MutexType4 & m4 , MutexType5 & m5 , MutexType6 & m6 , MutexType7 & m7 , MutexType8 & m8 , MutexType9 & m9 , MutexType10 & m10 , MutexType11 & m11 , MutexType12 & m12 , MutexType13 & m13 , MutexType14 & m14)
    {
        unsigned lock_first = 0;
        for (;;)
        {
            switch (lock_first)
            {
                case 0: lock_first = detail::lock_helper( m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13 , m14); if (!lock_first) return; break; case 1: lock_first = detail::lock_helper( m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13 , m14 , m0); if (!lock_first) return; lock_first = (lock_first+ 1) % 15; break; case 2: lock_first = detail::lock_helper( m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13 , m14 , m0 , m1); if (!lock_first) return; lock_first = (lock_first+ 2) % 15; break; case 3: lock_first = detail::lock_helper( m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13 , m14 , m0 , m1 , m2); if (!lock_first) return; lock_first = (lock_first+ 3) % 15; break; case 4: lock_first = detail::lock_helper( m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13 , m14 , m0 , m1 , m2 , m3); if (!lock_first) return; lock_first = (lock_first+ 4) % 15; break; case 5: lock_first = detail::lock_helper( m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13 , m14 , m0 , m1 , m2 , m3 , m4); if (!lock_first) return; lock_first = (lock_first+ 5) % 15; break; case 6: lock_first = detail::lock_helper( m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13 , m14 , m0 , m1 , m2 , m3 , m4 , m5); if (!lock_first) return; lock_first = (lock_first+ 6) % 15; break; case 7: lock_first = detail::lock_helper( m7 , m8 , m9 , m10 , m11 , m12 , m13 , m14 , m0 , m1 , m2 , m3 , m4 , m5 , m6); if (!lock_first) return; lock_first = (lock_first+ 7) % 15; break; case 8: lock_first = detail::lock_helper( m8 , m9 , m10 , m11 , m12 , m13 , m14 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7); if (!lock_first) return; lock_first = (lock_first+ 8) % 15; break; case 9: lock_first = detail::lock_helper( m9 , m10 , m11 , m12 , m13 , m14 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8); if (!lock_first) return; lock_first = (lock_first+ 9) % 15; break; case 10: lock_first = detail::lock_helper( m10 , m11 , m12 , m13 , m14 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9); if (!lock_first) return; lock_first = (lock_first+ 10) % 15; break; case 11: lock_first = detail::lock_helper( m11 , m12 , m13 , m14 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10); if (!lock_first) return; lock_first = (lock_first+ 11) % 15; break; case 12: lock_first = detail::lock_helper( m12 , m13 , m14 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11); if (!lock_first) return; lock_first = (lock_first+ 12) % 15; break; case 13: lock_first = detail::lock_helper( m13 , m14 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12); if (!lock_first) return; lock_first = (lock_first+ 13) % 15; break; case 14: lock_first = detail::lock_helper( m14 , m0 , m1 , m2 , m3 , m4 , m5 , m6 , m7 , m8 , m9 , m10 , m11 , m12 , m13); if (!lock_first) return; lock_first = (lock_first+ 14) % 15; break;
            }
        }
    }
}
