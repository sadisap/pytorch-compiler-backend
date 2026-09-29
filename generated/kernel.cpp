#include <algorithm>
#include <cstddef>
#include <vector>

extern "C" void run_kernel(
    const float* input,
    float* output,
    std::size_t n
) {
    std::vector<float> temp0(n);
    for (std::size_t i = 0; i < n; ++i) {
        temp0[i] = input[i] * 2.0f;
    }
    std::vector<float> temp1(n);
    for (std::size_t i = 0; i < n; ++i) {
        temp1[i] = temp0[i] + 3.0f;
    }
    std::vector<float> temp2(n);
    for (std::size_t i = 0; i < n; ++i) {
        temp2[i] = std::max(temp1[i], 0.0f);
    }

    for (std::size_t i = 0; i < n; ++i) {
        output[i] = temp2[i];
    }
}
