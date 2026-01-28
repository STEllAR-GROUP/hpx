#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include "./include/EasyBMP.h"
#include <vector>
#include <iostream>
#include <algorithm>

// Simple blur function for a single pixel row
void blur_row(const easy_bmp::BMP& input, easy_bmp::BMP& output, int y)
{
    int width = input.TellWidth();
    int height = input.TellHeight();

    for (int x = 0; x < width; ++x)
    {
        int r = 0, g = 0, b = 0, count = 0;

        // 3x3 kernel
        for (int ky = -1; ky <= 1; ++ky)
        {
            int ny = y + ky;
            if (ny < 0 || ny >= height) continue;

            for (int kx = -1; kx <= 1; ++kx)
            {
                int nx = x + kx;
                if (nx < 0 || nx >= width) continue;

                auto pixel = input(nx, ny);
                r += pixel->Red;
                g += pixel->Green;
                b += pixel->Blue;
                ++count;
            }
        }

        output(x, y)->Red   = r / count;
        output(x, y)->Green = g / count;
        output(x, y)->Blue  = b / count;
    }
}

// Full blur function for the image using par_simd
void blur_image(const easy_bmp::BMP& input, easy_bmp::BMP& output)
{
    int height = input.TellHeight();
    std::vector<int> rows(height);
    for (int i = 0; i < height; ++i) rows[i] = i;

    using int_simd = std::experimental::native_simd<int>;

    hpx::for_each(
        hpx::execution::par_simd,
        rows.begin(), rows.end(),
        [&](auto y_vec)  // generic lambda for par_simd
        {
            if constexpr (hpx::parallel::traits::is_vector_pack_v<decltype(y_vec)>) {
                // SIMD case
                for (size_t i = 0; i < y_vec.size(); ++i) {
                    int y = y_vec[i];
                    blur_row(input, output, y);
                }
            } else {
                // Scalar case
                int y = y_vec;
                blur_row(input, output, y);
            }
        }
    );
}
int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.bmp> <output.bmp>\n";
        return 1;
    }

    easy_bmp::BMP input;
    if (!input.ReadFromFile(argv[1])) {
        std::cerr << "Could not open " << argv[1] << "\n";
        return 1;
    }

    int width = input.TellWidth();
    int height = input.TellHeight();

    easy_bmp::BMP output;
    output.SetSize(width, height);
    output.SetBitDepth(24); // RGB

    // Copy input to output to preserve borders
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            *output(x, y) = *input(x, y);

    blur_image(input, output);

    output.WriteToFile(argv[2]);
    std::cout << "Blurred image written to " << argv[2] << "\n";

    return 0;
}

