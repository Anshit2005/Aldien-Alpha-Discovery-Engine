#include "CSVLoader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::vector<std::string> splitCSV(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;

    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }

    return fields;
}

double parseDouble(const std::string& value, const std::string& column) {
    try {
        size_t pos = 0;
        double result = std::stod(value, &pos);

        if (pos != value.size()) {
            throw std::runtime_error("Invalid numeric value");
        }

        return result;
    } catch (...) {
        throw std::runtime_error(
            "Failed to parse column '" + column + "': " + value
        );
    }
}

} // namespace

MarketData CSVLoader::load(const std::string& filename) {

    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Could not open CSV file: " + filename
        );
    }

    MarketData data;

    std::string line;

    // Read header
    if (!std::getline(file, line)) {
        throw std::runtime_error("CSV file is empty");
    }

    const std::vector<std::string> expectedHeader = {
        "datetime",
        "instrument",
        "open",
        "high",
        "low",
        "close",
        "volume"
    };

    std::vector<std::string> header = splitCSV(line);

    if (header != expectedHeader) {
        throw std::runtime_error(
            "Unexpected CSV header. Expected: "
            "datetime,instrument,open,high,low,close,volume"
        );
    }

    // Read rows
    size_t lineNumber = 1;

    while (std::getline(file, line)) {

        ++lineNumber;

        if (line.empty()) {
            continue;
        }

        std::vector<std::string> fields = splitCSV(line);

        if (fields.size() != 7) {
            throw std::runtime_error(
                "Invalid number of columns at CSV line " +
                std::to_string(lineNumber)
            );
        }

        data.datetime.push_back(fields[0]);
        data.instrument.push_back(fields[1]);

        data.open.push_back(
            parseDouble(fields[2], "open")
        );

        data.high.push_back(
            parseDouble(fields[3], "high")
        );

        data.low.push_back(
            parseDouble(fields[4], "low")
        );

        data.close.push_back(
            parseDouble(fields[5], "close")
        );

        data.volume.push_back(
            parseDouble(fields[6], "volume")
        );
    }

    if (data.size() == 0) {
        throw std::runtime_error("CSV contains no data rows");
    }

    return data;
}