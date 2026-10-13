#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct MarketData {

    std::vector<std::string> datetime;
    std::vector<std::string> instrument;

    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<double> volume;

    // Forward returns
    std::vector<double> forward_return_1d;
    std::vector<double> forward_return_5d;
    std::vector<double> forward_return_20d;
    std::vector<double> forward_return_30d;

    size_t size() const {
        return close.size();
    }
};