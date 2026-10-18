import Layout from "@theme/Layout";

import {Figure} from "../components/Figure";
import {ReferenceLink} from "../components/ReferenceLink";
import {Tex} from "../components/Tex";
import {siteUrl} from "../lib/site";

const REPOSITORY = "https://github.com/VitalyVorobyev/scorequant/tree/main";

/**
 * The research page: the one theorem the library relies on, where it stops,
 * and what in it is new. Static prose; the proofs and counterexamples it links
 * to are the record.
 */
export default function Research(): React.JSX.Element {
  return (
    <Layout
      title="Research"
      description="The theorem behind ScoreQuant's reusable rules, its limits, and what in it is new."
    >
      <article className="home-article">
        <header>
          <h1>Research</h1>
          <p className="home-article__lead">
            One theorem lets a fitted partition become a reusable rule. Its proof is machine-checked,
            and a few exact counterexamples mark where it stops.
          </p>
        </header>

        <section aria-labelledby="theorem">
          <h2 id="theorem">The theorem</h2>
          <p>
            Take a finite weighted sample of score vectors, split into cells, and let{" "}
            <Tex>I</Tex> be the Fisher information the cell counts keep. Call the partition stable
            if no single move of one event to another cell raises <Tex>{String.raw`\log\det I`}</Tex>.
            Then, when <Tex>I</Tex> is nonsingular, every event of a stable partition is strictly
            nearer the mean score of its own cell than to any other, in the metric of{" "}
            <Tex>{String.raw`I^{-1}`}</Tex>. A stable partition is therefore a Voronoi diagram in
            score space, and it compiles into a nearest-centroid rule that labels new events. Every
            globally optimal partition is stable, so the optimum is always such a rule. The{" "}
            <ReferenceLink to="guide/#the-one-crossing-compiling-a-partition">user guide</ReferenceLink>{" "}
            shows the call.
          </p>
          <Figure
            src={siteUrl("walkthrough-figures/michelson-d-geometry.png")}
            alt="Three panels. Top left: the raw score plane tinted by six convex cells with straight edges, the trajectory coloured by cell. Top right: the same cells after whitening, an ordinary Voronoi diagram of the cell means. Bottom: the same labels along the detector, a comb of many intervals above six equal segments."
            caption="A stable partition from the Michelson walkthrough. After whitening by the binned information (right), its cells are an ordinary Voronoi diagram of the cell means."
            paper
          />
        </section>

        <section aria-labelledby="limits">
          <h2 id="limits">Where it stops</h2>
          <p>
            <strong>Lloyd iteration can go backwards.</strong> Moving every event to its nearest
            centroid in the current metric, all at once, can lower{" "}
            <Tex>{String.raw`\log\det I`}</Tex>. The library keeps such a step only when the exact
            objective rises.
          </p>
          <p>
            <strong>A Voronoi partition need not be stable.</strong> The converse fails: a
            partition can be nearest-centroid in that metric and still admit a single move that
            raises <Tex>{String.raw`\log\det I`}</Tex>.
          </p>
          <p>
            <strong>Nuisance parameters break the geometry.</strong> When nuisance parameters are
            profiled out (<Tex>{String.raw`D_s`}</Tex>), even a globally optimal partition can
            violate its own nearest-cell rule. Such a partition does not compile, and a rule has to
            be fitted directly.
          </p>
        </section>

        <section aria-labelledby="new">
          <h2 id="new">What is new</h2>
          <p>
            The exact change in the determinant when one event moves is classical: Banfield and
            Bassill (1977) used it in a transfer algorithm for clustering, and Fedorov-style exchange
            in optimal design rests on the same algebra. For a single parameter, binning by the
            score is score-function quantization (Venkitasubramaniam, Tong and Swami, 2007). The new
            part is the statement above, for the determinant criterion and a vector parameter.
          </p>
        </section>

        <section aria-labelledby="record">
          <h2 id="record">Proofs and counterexamples</h2>
          <p>
            The theorem and the lemmas under it are proved in Lean 4 with Mathlib, in{" "}
            <a href={`${REPOSITORY}/research/formal`}>research/formal</a>. The proofs check the
            mathematics, not the Python code. Each limit above is an exact fixture in{" "}
            <a href={`${REPOSITORY}/tests/fixtures/counterexamples`}>
              tests/fixtures/counterexamples
            </a>
            , and the library names the fixture when it refuses a request. A short paper is in
            preparation.
          </p>
        </section>
      </article>
    </Layout>
  );
}
