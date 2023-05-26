namespace hpx::experimental { namespace ranges {
    template <typename Iter, typename T>
    struct in_value_result
    {
        [[no_unique_address]] Iter in;
        [[no_unique_address]] T value;

        template <class I2, class T2>
        // hpx::convertible?
        constexpr operator in_value_result<I2, T2>() const&
        {
            return {in, value};
        }

        template <class I2, class T2>
        // hpx::convertible?
        constexpr operator in_value_result<I2, T2>() &&
        {
            return {std::move(in), std::move(value)};
        }
    };
}}    // namespace hpx::experimental::ranges