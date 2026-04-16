# 1. Project Overview

The goal of the project is to apply and compare different AI techniques covered in the course to the same problem.

# 2. Project Description

## Problem Formulation

Each group must define a problem that can be addressed using techniques covered in
the course. The problem should be sufficiently complex to allow meaningful comparison
between different approaches.

The problem description should include:

- A clear description of the task or objective.
- A formal representation of the problem where appropriate (e.g., state space, variables, constraints, probabilities).
- An explanation of why the problem is suitable for multiple AI techniques covered in the course.

## Implementation of Multiple Solution Approaches

Each group must implement three different solution approaches to the same problem,
based on different techniques or algorithms covered in the course. Note that all
implementations should in general solve the same underlying problem so that
meaningful comparisons can be made.

Approaches may differ in their modeling assumptions, representations, or algorithmic
strategies. For example, differences may concern:


- probabilistic vs. deterministic models;
- model-based vs. learning-based approaches;
- inclusion of responsible AI techniques.

Note: While Python is the preferred language for these implementations, alternative tools
or programming languages may be used if accompanied by a clear justification. Please
note that if high-level packages are employed, a thorough understanding of their
underlying mechanisms is expected.

## Evaluation and Comparison


The project must include both experimental evaluation and qualitative comparison of
the implemented approaches. The goal is not only to implement different methods, but
also to analyze and understand the trade-offs between them.

Experimental evaluation may include:


- solution quality or accuracy;
- runtime or computational cost;
- scalability as problem size increases;
- performance across different problem instances.

Qualitative comparison may discuss:


- strengths and weaknesses of each method;
- the impact of modeling assumptions;
- situations in which one approach may be preferable to others.

Results should be presented clearly using tables, figures, or plots where appropriate.

# 3. Evaluation Rubric

The project will be graded out of 10 points, according to the following rubric.

```
Aspect Excellent (A) Good (B)
Satisfactory
(C)
```
```
Problem
Formulation (
pts)
```
```
Problem is clearly
defined and well-
motivated; includes a
precise formalization;
problem is clearly
suitable for multiple AI
techniques.
```
```
Problem is generally
clear but the
formalization is
somewhat
incomplete or lacks
precision; suitability
for multiple
approaches is
present but not fully
justified.
```
```
Problem
description is
vague or
partially
specified;
formalization is
minimal; limited
justification for
why multiple AI
methods apply.
```
```
Methodological
Approaches (
pts)
```
```
Three clearly distinct
approaches based on
course techniques;
each method is
correctly implemented
and meaningfully
different.
```
```
Three approaches
implemented but
differences are
limited or some
implementations
lack depth or clarity.
```
```
Approaches are
only
superficially
different,
partially
implemented, or
not clearly
connected to
course
techniques.
```

```
Experimental
Evaluation (
pts)
```
```
Well-designed
experiments with
appropriate metrics;
results presented
clearly; experiments
allow meaningful
comparison across
methods.
```
```
Experiments
conducted but
somewhat limited in
scope, number of
instances, or
evaluation metrics.
```
```
Minimal
experimentation;
results reported
but evaluation
design is weak
or incomplete.
```
```
Qualitative
Comparison (
pts)
```
```
Insightful discussion
of trade-offs; clearly
explains strengths and
weaknesses of each
method; discusses
modeling assumptions
and when one method
would be preferable.
```
```
Some comparison
between
approaches;
strengths and
weaknesses
mentioned but
discussion is
somewhat
superficial.
```
```
Limited
comparison;
mostly
descriptive with
little analysis of
trade-offs or
assumptions.
```
```
Te c h n i c a l
Quality (1 pt)
```
```
Code is well-
structured, readable,
and reproducible;
implementations run
correctly and
experiments can be
replicated.
```
```
Code generally
works but
organization, clarity,
or reproducibility
could be improved.
```
```
Code difficult to
follow, partially
incomplete, or
difficult to
reproduce
results.
```
```
Report Quality
(1 pt)
```
```
Report is clear, well-
organized, and
concise; includes
appropriate
figures/tables and
clearly explains
methods and results.
```
```
Report generally
understandable but
some sections lack
clarity or
organization.
```
```
Report difficult
to follow;
explanations
incomplete or
poorly
structured.
```
# 4. Submission Instructions

Submit your project via UV by May 17 as a single PDF report of up to 4 pages following
the [AAAI 2026 formatting instructions](https://aaai.org/conference/aaai/aaai-26/submission-instructions/).

Note:


- References may appear on additional pages.
- The report must include a link to a GitHub repository containing the project code and clear instructions for reproducing the results presented in the report.
- During the final week, you will present your work via a short presentation.

# 5. A Concrete Example

Here's an example of a simple project that illustrates the different elements requested.

## Problem Formulation

A robot must navigate a grid-based warehouse to pick up items and deliver them to a
drop-off point. However, the warehouse may have "dynamic obstacles" (other robots or
moving workers) and "uncertainty" (some paths might be blocked or slippery).

### Formal Representation:


- State Space (S): Coordinates of the robot (x,y), battery level, and inventory status.
- Actions (A): Move {N, S, E, W}, Pick-up, Drop-off.
- Constraints: Avoid static walls and dynamic obstacles; finish before battery reaches 0.
- Probabilities: P(s'|s,a) representing the probability of successfully moving to a cell vs. slipping/staying in place.

## Methodological Approaches

### Approach A: Search


- Logic: Treat the warehouse as a static graph. Use Manhattan distance as a heuristic.
- Assumption: The world is known and static. If an obstacle moves into the path, the robot must "re-plan" from scratch.

### Approach B: Constraint Satisfaction Problem (CSP)


- Logic: Model the path as a sequence of variables X_t (position at time t). Define constraints such that X_t and X_{t+1} must be adjacent and X_t cannot be an obstacle.
- Assumption: Useful for finding any feasible path that satisfies complex business rules (e.g., "cannot carry Item A and Item B simultaneously").

### Approach C: Markov Decision Processes (MDP)


- Logic: Calculate a "policy" \pi(s) that tells the robot the best move for every possible square, accounting for the 10% chance that a motor slip occurs.
- Assumption: The environment is stochastic. The goal is to maximize the expected long-term reward.

## Experimental Evaluation

The approaches were tested across 100 simulated runs with varying obstacle density.

```
Metric
Approach A
(Search)
Approach B (CSP)
Approach C
(MDP)
Path
Optimality
Highest (Shortest) Variable (Feasible) High (Expected)
```
```
Execution
Time
<10ms ~50ms
High (Offline
prep)
```
```
Robustness Low (Fails on slip)
Medium (Re-
calculates)
Highest
```
```
Scalability High Low Moderate
```

## Qualitative Comparison


- Strengths: Search is incredibly fast for simple navigation. CSP excels when "business rules" (e.g., "don't carry hazardous and flammable items together") are the priority. MDPs are the only choice for truly unpredictable environments where failure is costly.
- Weaknesses: Search is too "brittle" for real-world movement. CSPs become computationally expensive as the time horizon (t) increases. MDPs suffer from the "curse of dimensionality" in very large warehouses.
- Conclusion: For a standard warehouse, a hybrid approach, i.e., using Search for long-distance routing and an MDP policy for local obstacle avoidance, would likely yield the best real-world results.
