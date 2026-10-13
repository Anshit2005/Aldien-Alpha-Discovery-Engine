#include "CSVLoader.h"

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> splitCSV(
    const std::string& line
) {
    std::vector<std::string> fields;

    std::stringstream ss(line);
    std::string field;

    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }

    // std::getline() does not preserve a trailing empty field.
    // Example:
    // "a,b," → ["a", "b"]
    //
    // We need:
    // "a,b," → ["a", "b", ""]
    if (!line.empty() &&
        line.back() == ',') {

        fields.emplace_back();
    }

    return fields;
}


double parseDouble(
    const std::string& value,
    const std::string& column
) {
    // pandas writes missing floating-point values as "nan".
    if (value.empty() ||
        value == "nan" ||
        value == "NaN" ||
        value == "NAN") {

        return std::numeric_limits<double>::quiet_NaN();
    }

    try {

        size_t pos = 0;

        double result =
            std::stod(value, &pos);

        if (pos != value.size()) {
            throw std::runtime_error(
                "Invalid numeric value"
            );
        }

        return result;

    } catch (...) {

        throw std::runtime_error(
            "Failed to parse column '" +
            column +
            "': " +
            value
        );
    }
}

} // namespace


MarketData CSVLoader::load(
    const std::string& filename
) {

    std::ifstream file(filename);

    if (!file.is_open()) {

        throw std::runtime_error(
            "Could not open CSV file: " +
            filename
        );
    }

    MarketData data;

    std::string line;

    // ------------------------------------------------------------
    // Header
    // ------------------------------------------------------------

    if (!std::getline(file, line)) {

        throw std::runtime_error(
            "CSV file is empty"
        );
    }

    const std::vector<std::string>
        expectedHeader = {

            "datetime",
            "instrument",
            "open",
            "high",
            "low",
            "close",
            "volume",

            "forward_return_1d",
            "forward_return_5d",
            "forward_return_20d",
            "forward_return_30d"
        };

    const std::vector<std::string>
        header = splitCSV(line);

    if (header != expectedHeader) {

        throw std::runtime_error(
            "Unexpected CSV header"
        );
    }

    // ------------------------------------------------------------
    // Rows
    // ------------------------------------------------------------

    size_t lineNumber = 1;

    while (std::getline(file, line)) {

        ++lineNumber;

        if (line.empty()) {
            continue;
        }

        std::vector<std::string>
            fields = splitCSV(line);

        if (fields.size() != 11) {

            throw std::runtime_error(
                "Invalid number of columns at "
                "CSV line " +
                std::to_string(lineNumber)
            );
        }

        data.datetime.push_back(
            fields[0]
        );

        data.instrument.push_back(
            fields[1]
        );

        data.open.push_back(
            parseDouble(
                fields[2],
                "open"
            )
        );

        data.high.push_back(
            parseDouble(
                fields[3],
                "high"
            )
        );

        data.low.push_back(
            parseDouble(
                fields[4],
                "low"
            )
        );

        data.close.push_back(
            parseDouble(
                fields[5],
                "close"
            )
        );

        data.volume.push_back(
            parseDouble(
                fields[6],
                "volume"
            )
        );

        data.forward_return_1d.push_back(
            parseDouble(
                fields[7],
                "forward_return_1d"
            )
        );

        data.forward_return_5d.push_back(
            parseDouble(
                fields[8],
                "forward_return_5d"
            )
        );

        data.forward_return_20d.push_back(
            parseDouble(
                fields[9],
                "forward_return_20d"
            )
        );

        data.forward_return_30d.push_back(
            parseDouble(
                fields[10],
                "forward_return_30d"
            )
        );
    }

    if (data.size() == 0) {

        throw std::runtime_error(
            "CSV contains no data rows"
        );
    }

    return data;
}