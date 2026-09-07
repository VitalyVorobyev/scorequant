import {factsFor} from "../lib/facts";

/**
 * The walkthrough index: one card per applied problem.
 *
 * A card states the problem first, then the data, and only then the parts of
 * the library the page uses, as tags a reader can match against their own
 * situation. The task tag is always present, because whether a page labels a
 * fixed table or fits a reusable rule is the first thing to know about it. Bin
 * budgets come from the walkthrough facts, never from a literal typed here.
 */

export type TagKind = "task" | "input" | "criterion" | "solver";

export interface WalkthroughTag {
  kind: TagKind;
  label: string;
}

export interface WalkthroughCard {
  slug: string;
  title: string;
  /** Route within the portal. */
  href: string;
  /** The question the page answers, in one sentence; shown first. */
  lead: string;
  /** What is measured and what is to be estimated, and what the page finds. */
  problem: string;
  /** Where the data comes from and what kind of evidence it is. */
  data: string;
  tags: readonly WalkthroughTag[];
}

export const TAG_KIND_LABELS: Record<TagKind, string> = {
  task: "Task",
  input: "Input",
  criterion: "Criterion",
  solver: "Solver"
};

const michelson = factsFor("michelson");
const ratios = factsFor("ratios");
const flowcyt = factsFor("flowcyt");
const hep = factsFor("hep");

export const WALKTHROUGHS: readonly WalkthroughCard[] = [
  {
    slug: "michelson",
    title: "Phase Estimation in a Michelson Interferometer",
    href: "/walkthroughs/michelson",
    lead:
      `Which detector positions should ${michelson("bins")} counters collect from, so the fringe ` +
      "phase survives while its frequency floats?",
    problem:
      "A fringe pattern is read out by a detector with a fixed number of counters, and the " +
      "fringe phase is to be estimated while the fringe frequency floats beside it. The page " +
      `asks which detector positions each of ${michelson("bins")} counters should collect from, ` +
      "and shows why equal segments can lose the phase entirely.",
    data:
      "An idealised photon-arrival model with a closed-form score, integrated by deterministic " +
      "quadrature. No recorded instrument data.",
    tags: [
      {kind: "task", label: "optimize_partition"},
      {kind: "task", label: "fit_quantizer"},
      {kind: "input", label: "ScoreFunction"},
      {kind: "input", label: "IntegrationSource"},
      {kind: "criterion", label: "ProfiledDOptimality"},
      {kind: "criterion", label: "DOptimality"},
      {kind: "solver", label: "DExchangeConfig"},
      {kind: "solver", label: "SoftVoronoiConfig"}
    ]
  },
  {
    slug: "ratios",
    title: "From a classifier to Fisher-preserving bins",
    href: "/walkthroughs/ratios",
    lead:
      "Can a calibrated classifier stand in for the likelihood, and is the retention it reports " +
      "about the data or about the classifier?",
    problem:
      "A two-component mixture whose signal fraction is to be estimated, with the component " +
      "densities standing in for a likelihood nobody has. A classifier's calibrated posteriors, " +
      "divided by its training proportions, estimate the density ratio the score depends on. The " +
      `page fits ${ratios("bins")} bins on that estimate and asks whether the retention the ` +
      "library reports is about the data or about the classifier.",
    data:
      "A synthetic mixture of two normal densities, generated so that the exact score is also " +
      "known and the estimate can be checked against it.",
    tags: [
      {kind: "task", label: "fit_quantizer"},
      {kind: "input", label: "DensityRatioScore"},
      {kind: "criterion", label: "DOptimality"},
      {kind: "solver", label: "DExchangeConfig"}
    ]
  },
  {
    slug: "hep",
    title: "A Higgs search with a floating energy scale",
    href: "/walkthroughs/hep",
    lead:
      `Fit a ${hep("bins")}-category rule on half the simulated collisions, apply it to the other ` +
      "half, and measure the signal strength while the tau energy scale floats.",
    problem:
      "Simulated collision events, a signal strength to estimate, and a tau energy scale that " +
      `floats in the fit as a nuisance. Only ${hep("bins")} counts are kept. The page fits a ` +
      "reusable rule for the profiled information about the signal strength, applies it to " +
      "events it never saw, and shows in the fitted uncertainty why slices of the classifier " +
      "output fall apart once the energy scale floats.",
    data:
      "Simulated events from the FAIR Universe HiggsML public dataset, each with a Monte Carlo " +
      "weight and shifted-energy-scale copies. A simulation study, not an analysis.",
    tags: [
      {kind: "task", label: "fit_quantizer"},
      {kind: "task", label: "optimize_partition"},
      {kind: "input", label: "ScoreSample"},
      {kind: "criterion", label: "ProfiledDOptimality"},
      {kind: "criterion", label: "DOptimality"},
      {kind: "solver", label: "SoftVoronoiConfig"},
      {kind: "solver", label: "DExchangeConfig"}
    ]
  },
  {
    slug: "flowcyt",
    title: "Bone-marrow cell populations",
    href: "/walkthroughs/flowcyt",
    lead:
      `Fit ${flowcyt("bins")} categories on reference patients, apply them to held-out patients, ` +
      "and measure what the reduction costs.",
    problem:
      "Each cell carries marker intensities and an expert label; the quantity to estimate is a " +
      `patient's population fractions. The page fits ${flowcyt("bins")} categories on reference ` +
      "patients from classifier-estimated scores, freezes them, applies them to held-out " +
      "patients, fits the fractions from the counts, and measures the error the reduction costs.",
    data:
      "Real cells from the FlowCyt classification benchmark, with whole patients held out before " +
      "anything was fitted.",
    tags: [
      {kind: "task", label: "fit_quantizer"},
      {kind: "input", label: "ScoreSample"},
      {kind: "criterion", label: "DOptimality"},
      {kind: "solver", label: "SoftVoronoiConfig"},
      {kind: "solver", label: "DExchangeConfig"}
    ]
  }
];
