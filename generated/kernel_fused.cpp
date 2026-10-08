#include <algorithm>
#include <cstddef>
#include <vector>

extern "C" void run_kernel(
    const float* input,
    float* output,
    std::size_t n
) {
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = ((std::max(((input[i] * 0x1.0000000000000p+1f) + 0x1.8000000000000p+1f), 0.0f) * 0x1.0000000000000p-1f) + 0x1.0000000000000p+0f);
    }
}
