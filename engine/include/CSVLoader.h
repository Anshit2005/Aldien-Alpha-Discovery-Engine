#pragma once

#include "Data.h"
#include <string>

class CSVLoader {
public:
    static MarketData load(const std::string& filename);
};