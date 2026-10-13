#pragma once

#include <vector>
#include <string>
#include <cstddef>

struct MarketData {
    std::vector<std::string> datetime;
    std::vector<std::string> instrument;

    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<double> volume;

    size_t size() const {
        return close.size();
    }
};