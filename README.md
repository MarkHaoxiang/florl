# Florl (*floral*)

florl is an open-source framework to develop federated reinforcement learning (FRL) applications.

Our framework builds upon [Flower](https://github.com/adap/flower) and [TorchRL](https://github.com/pytorch/rl). 

This project is in active development. Check out a [report](https://github.com/MarkHaoxiang/florl/blob/l361-submission/report/fl_florl_report.pdf) that detailed a past version and the original motivations for this project.

Currently, this project is in active development and not yet functional. However, branch `l361-submission` contains a suite of FRL experiments that may be useful for reference.

## Installation

## Abstractions

`florl` provides an opinionated research pipeline for FRL.

### Task

A Task represents a challenging scenario in FRL, and generally contains the environment / dataset, associated constraints for the solution space, hyper-parameters etc.. A Task should contain a set of associated evaluations and visualisations. Technically, a Task is an interface with hooks for Algorithm.

### Experiment

An Experiment is instantiated from a task and an algorithm which hooks in the task. Experiment is the primary entry point for executing a research workflow, and manages the flwr server / flwr client / evaluation orchestration.

### Algorithm

An Algorithm is a dependency injection into Task, which contains components and logic to solve the Task. Algorithms can have additional associated metrics.

### Benchmark

A benchmark is a collection of experiments, which share the same Task. Benchmarks can be grouped.\

### Development

We use [uv](https://github.com/astral-sh/uv) to manage local development environments, and provide a set of .vscode settings.

`uv sync --all-extras'