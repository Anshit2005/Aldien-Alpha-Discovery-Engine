# Aldien — Alpha Discovery Engine

LLM-guided evolutionary alpha discovery with distributed C++ execution and statistical validation.

## Overview

Aldien is a research-oriented alpha discovery system designed to automatically generate, evaluate, evolve, and statistically validate quantitative trading factors.

The system combines:

- LLM-guided symbolic alpha generation
- Evolutionary search
- A message-queue-based distributed execution architecture
- A high-performance C++ alpha evaluation engine
- Statistical validation and robustness testing

The objective is to bridge symbolic hypothesis generation with scalable and rigorous quantitative validation.

## Architecture

```text
Market Data
     │
     ▼
Universe & Data Preparation
     │
     ▼
LLM-Guided Alpha Generation
     │
     ▼
Evolutionary Search
     │
     ▼
Message Queue
     │
     ▼
Distributed C++ Execution Engine
     │
     ▼
Factor Performance Evaluation
     │
     ▼
Statistical Validation
     │
     ▼
Surviving Alpha Factors