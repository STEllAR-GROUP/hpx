#include <hpx/iterator_support/tests/iter_sent.hpp>
#include <hpx/modules/testing.hpp>

namespace test {
    ///////////////////////////////////////////////////////////////////////////
    template <typename IterType>
        requires(hpx::traits::is_iterator<IterType>::value)
    struct sentinel_from_iterator
    {
        explicit sentinel_from_iterator(IterType end_iter)
          : end(end_iter)
        {
        }

        IterType get()
        {
            return end;
        }

        friend bool operator==(IterType i, sentinel_from_iterator<IterType> s)
        {
            return i == s.get();
        }

        friend bool operator==(sentinel_from_iterator<IterType> s, IterType i)
        {
            return i == s.get();
        }

        friend bool operator!=(IterType i, sentinel_from_iterator<IterType> s)
        {
            return i != s.get();
        }

        friend bool operator!=(sentinel_from_iterator<IterType> s, IterType i)
        {
            return i != s.get();
        }

        friend auto operator-(sentinel_from_iterator<IterType> s, IterType i)
            requires hpx::traits::is_random_access_iterator_v<IterType>
        {
            return s.get() - i;
        }

        friend auto operator-(IterType i, sentinel_from_iterator<IterType> s)
            requires hpx::traits::is_random_access_iterator_v<IterType>
        {
            return i - s.get();
        }

    private:
        IterType end;
    };
}    // namespace test