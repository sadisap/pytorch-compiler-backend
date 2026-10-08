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
        temp0[i] = input[i] * 0x1.0000000000000p+1f;
    }
    std::vector<float> temp1(n);
    for (std::size_t i = 0; i < n; ++i) {
        temp1[i] = temp0[i] + 0x1.8000000000000p+1f;
    }
    std::vector<float> temp2(n);
    for (std::size_t i = 0; i < n; ++i) {
        temp2[i] = std::max(temp1[i], 0.0f);
    }
    std::vector<float> temp3(n);
    for (std::size_t i = 0; i < n; ++i) {
        temp3[i] = temp2[i] * 0x1.0000000000000p-1f;
    }
    std::vector<float> temp4(n);
    for (std::size_t i = 0; i < n; ++i) {
        temp4[i] = temp3[i] + 0x1.0000000000000p+0f;
    }
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = temp4[i];
    }
}
