// Research scalar packed-weight matrix product, not a production GGML kernel.
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <limits>
#include <climits>

static float half_le(const uint8_t* p) {
    const unsigned u = unsigned(p[0]) | (unsigned(p[1]) << 8);
    const unsigned e = (u >> 10) & 31, m = u & 1023;
    float x;
    if (e == 31) return std::numeric_limits<float>::quiet_NaN();
    if (e == 0) x = std::ldexp(float(m), -24);
    else x = std::ldexp(float(1024 + m), int(e) - 25);
    return (u & 32768) ? -x : x;
}
static int signed_byte(uint8_t x) { return x < 128 ? int(x) : int(x) - 256; }

extern "C" int r23_packed_linear(
    int type, const uint8_t* weights, size_t weight_bytes,
    const float* input, size_t input_elements,
    float* output, size_t output_elements,
    size_t rows, size_t cols, size_t batch) {
    if ((type != 4 && type != 8) || !weights || !input || !output ||
        !rows || !cols || !batch || cols % 32) return 1;
    const size_t block_bytes = type == 4 ? 18 : 34;
    const size_t blocks = cols / 32;
    if (blocks > SIZE_MAX / block_bytes) return 1;
    const size_t stride = blocks * block_bytes;
    if (rows > SIZE_MAX / stride || batch > SIZE_MAX / cols || batch > SIZE_MAX / rows ||
        weight_bytes != rows * stride || input_elements != batch * cols ||
        output_elements != batch * rows) return 1;
    // Validate before writing outputs. No scratch weight matrix is allocated.
    for (size_t i = 0; i < input_elements; ++i)
        if (!std::isfinite(input[i])) return 2;
    for (size_t i = 0; i < rows * blocks; ++i)
        if (!std::isfinite(half_le(weights + i * block_bytes))) return 2;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t r = 0; r < rows; ++r) {
            double sum = 0.;
            for (size_t k = 0; k < blocks; ++k) {
                const uint8_t* p = weights + r * stride + k * block_bytes;
                const double d = half_le(p);
                const float* x = input + b * cols + k * 32;
                if (type == 4) {
                    for (size_t j = 0; j < 16; ++j) {
                        sum += d * (int(p[2+j] & 15) - 8) * double(x[j]);
                        sum += d * (int(p[2+j] >> 4) - 8) * double(x[j+16]);
                    }
                } else {
                    for (size_t j = 0; j < 32; ++j)
                        sum += d * signed_byte(p[2+j]) * double(x[j]);
                }
            }
            if (!std::isfinite(sum) || std::abs(sum) > std::numeric_limits<float>::max()) return 3;
            output[b * rows + r] = float(sum);
        }
    }
    return 0;
}
