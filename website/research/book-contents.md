---
title: The book, chapter by chapter
sidebar_label: The book
sidebar_position: 8
---

import {ReferenceLink} from "@site/src/components/ReferenceLink";

# The book, chapter by chapter

The reference carries a fourteen-chapter derivation that starts from a one-parameter example
worked by hand and ends with a decision table. It is written to be read in order, but each chapter
states the question it answers, so it also works as a lookup. The chapters live in the reference
site; the
<ReferenceLink to="book/">book overview page</ReferenceLink> is their front matter.

Chapters 1 to 4 are the setup, 5 to 9 are the determinant theory this section's
[what ScoreQuant adds](/research/what-scorequant-adds) page summarizes, 10 to 13 are the
extensions — nuisance parameters, the criterion that was rejected, differentiable rules, estimated
scores — and 14 is the map.

| # | Chapter | The question it answers | Read it if |
| --- | --- | --- | --- |
| 1 | <ReferenceLink to="book/ch01-why-bin/">Why bin at all</ReferenceLink> | Why analyses end in a handful of counts, and what that costs | You are deciding whether this problem is your problem |
| 2 | <ReferenceLink to="book/ch02-one-dimension/">One dimension by hand</ReferenceLink> | Where the two-cell boundaries go for a Gaussian, and why the answer is 2/π | You want the whole structure in miniature before any matrices appear |
| 3 | <ReferenceLink to="book/ch03-exact-1d/">Exact 1D binning by dynamic programming</ReferenceLink> | Why one-dimensional cells are intervals, and how to get the exact optimum rather than a fixed point | You have one score coordinate, or you want an exactly solvable case to check against |
| 4 | <ReferenceLink to="book/ch04-scores-and-doors/">Scores, score laws, and the three doors</ReferenceLink> | What a score is, and the three ways one actually reaches you — supplied, from densities, from ratios | You are not sure which input route your problem takes |
| 5 | <ReferenceLink to="book/ch05-information-after-binning/">Information after hard labels</ReferenceLink> | Exactly how much Fisher information the counts carry, in one exact formula | You want the single identity everything else is built on |
| 6 | <ReferenceLink to="book/ch06-two-tasks/">Two tasks and three optimization levels</ReferenceLink> | Why labelling a sample and building a reusable rule are different problems with different result types | You expected one `fit` and found two entry points |
| 7 | <ReferenceLink to="book/ch07-trace-kmeans/">The trace criterion and whitened k-means</ReferenceLink> | Why maximizing retained trace *is* weighted k-means, and the two places that stops being enough | You are wondering whether plain k-means would have done |
| 8 | <ReferenceLink to="book/ch08-d-optimality/">D-optimality and exact exchange</ReferenceLink> | Why the determinant is the criterion of choice, and the exact relocation algebra that makes it solvable | You want the core theory: the confidence-ellipsoid argument and the exchange solver |
| 9 | <ReferenceLink to="book/ch09-mahalanobis-lloyd/">Mahalanobis geometry and guarded Lloyd</ReferenceLink> | Why the obvious batch algorithm can make the objective worse, and what guard fixes it | You want a fast solver and need to know what the guard is protecting you from |
| 10 | <ReferenceLink to="book/ch10-profiled-ds/">Nuisance parameters and profiled D_s</ReferenceLink> | How to spend cells on the parameters you will actually report | Some of your parameters are calibration constants you will marginalize away |
| 11 | <ReferenceLink to="book/ch11-e-optimality/">E-optimality: why not</ReferenceLink> | Why the worst-direction criterion is named and then not implemented | You wanted a guarantee for every parameter direction, not for the volume |
| 12 | <ReferenceLink to="book/ch12-soft-rules/">Soft rules, purification, and consistency</ReferenceLink> | Why gradient descent on a hard objective is not an algorithm, and what a randomized rule buys instead | You are choosing between the exact and the differentiable solver |
| 13 | <ReferenceLink to="book/ch13-estimated-scores/">Estimated density ratios and scores</ReferenceLink> | What changes when the score is learned rather than computed | Your score comes from a classifier, a simulator, or a fitted ratio |
| 14 | <ReferenceLink to="book/ch14-choosing-a-method/">Diagnostics and choosing a method</ReferenceLink> | Which criterion, which solver, which diagnostics — and an explicit list of what remains unsolved | You have read enough and want the decision table |

Chapter 14's decision table is executed as part of the test suite: it enumerates every
criterion-and-configuration pairing and checks that each is accepted or refused as documented, so
the table in the book cannot drift away from the library's behaviour.

## Next

[Reading the claim record](/research/reading-the-claim-record) explains how to check any statement
in those chapters against its registry entry.
