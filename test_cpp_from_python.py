import pandas as pd

from python.evaluation.cpp_engine import CppAlphaEngine


# ------------------------------------------------------------
# Load development dataset
# ------------------------------------------------------------

DATA_PATH = (
    "data/processed/cpp_fitness_data.csv"
)

data = pd.read_csv(
    DATA_PATH,
    parse_dates=["datetime"]
)


# ------------------------------------------------------------
# Create C++ engine
# ------------------------------------------------------------

engine = CppAlphaEngine(data)


# ------------------------------------------------------------
# Alpha expression
# ------------------------------------------------------------

expression = (
    "RANK("
    "DIV("
    "SUB(CLOSE,TS_MEAN(CLOSE,20)),"
    "TS_STD(CLOSE,20)"
    ")"
    ")"
)


# ------------------------------------------------------------
# Evaluate fitness
# ------------------------------------------------------------

fitness = engine.evaluate_fitness_expression(
    expression,
    horizon=20
)


# ------------------------------------------------------------
# Print result
# ------------------------------------------------------------

print()
print("========================================")
print("      C++ FITNESS FROM PYTHON")
print("========================================")

print(f"Expression: {expression}")
print()

print(f"Mean IC:            {fitness.mean_ic}")
print(f"IC Std:             {fitness.ic_std}")
print(f"ICIR:               {fitness.icir}")
print(f"Number of IC days:  {fitness.num_days}")
print(f"Valid observations: {fitness.num_observations}")
print(f"Coverage:           {fitness.coverage}")

print("========================================")