import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";

import {Tex} from "../components/Tex";

/**
 * The portal's front page: what the library is for, stated before any API
 * name, and the three places to go next. It quotes no measurement
 * (`tests/test_walkthrough_facts.py` holds it to that) and runs nothing.
 */
export default function Home(): React.JSX.Element {
  return (
    <Layout
      title="ScoreQuant"
      description="Hard bins, with a measurement of what the binning cost: information-preserving quantization in score space."
    >
      <article className="home-article">
        <header>
          <h1>ScoreQuant</h1>
          <p className="home-article__lead">
            <strong>Hard bins, with a measurement of what the binning cost.</strong>
          </p>
          <p>
            You have a continuous observation, but downstream you need a small number of hard
            categories.
          </p>
          <p>
            How should you choose them without throwing away the information needed to estimate
            your parameters?
          </p>
          <p>
            ScoreQuant does the binning in <strong>score space</strong> and reports how much
            Fisher information survived.
          </p>
          <p className="home-article__install">
            <code>uv add scorequant</code>
          </p>
          <p>
            <Link to="/get-started">Get started →</Link>
          </p>
        </header>

        <section aria-labelledby="problem">
          <h2 id="problem">The problem</h2>
          <p>
            Suppose an observation <Tex>x</Tex> depends on parameters{" "}
            <Tex>{String.raw`\theta`}</Tex>, and you ultimately need to replace it by one of{" "}
            <Tex>K</Tex> labels:
          </p>
          <Tex display>{String.raw`x \longrightarrow \{1,\ldots,K\}.`}</Tex>
          <p>Many different binnings are possible. They are not equally useful for inference.</p>
          <p>
            For local parameter estimation, the relevant representation of an observation is its{" "}
            <strong>score</strong>
          </p>
          <Tex display>{String.raw`s(x)=\nabla_\theta \log p(x\mid\theta)\big|_{\theta_0}.`}</Tex>
          <p>ScoreQuant searches for hard cells in this score space:</p>
          {/* Every step plain: `.showcase-pipeline strong` is the accent the site
              uses for the step ScoreQuant itself performs, and this chain is the
              data, not the call. The get-started page marks `fit_quantizer`. */}
          <div className="showcase-pipeline">
            <span>observations</span>
            <span>scores</span>
            <span>K hard labels</span>
            <span>information preserved</span>
          </div>
          <p>
            The result is not only a partition. ScoreQuant also measures the information retained
            by that partition, and reports it as a D-efficiency: a dimensionless ratio, close to
            one when the labels kept nearly all of the Fisher information and small when most of
            it was spent on the compression.
          </p>
        </section>

        <section aria-labelledby="why">
          <h2 id="why">Why score space?</h2>
          <p>
            Two observations may look very different in the original observation space while having
            almost the same effect on parameter estimation. Conversely, nearby observations may
            carry quite different parameter information.
          </p>
          <p>The score removes that distinction.</p>
          <p>
            Binning loses information when score vectors that matter differently for inference are
            forced into the same cell. This makes score space the natural place to design the
            cells.
          </p>
          <p>ScoreQuant optimizes that compression directly.</p>
        </section>

        <section aria-labelledby="tasks">
          <h2 id="tasks">Two ways to use it</h2>

          <h3>Fit a reusable quantizer</h3>
          <p>Usually, you need a rule that can also label future observations.</p>
          <pre className="home-article__code">
            <code>
              {"quantizer = sq.fit_quantizer(...)\nlabels = quantizer.predict_scores(new_scores)"}
            </code>
          </pre>
          <p>
            The fitted quantizer defines hard cells in score space and applies the same rule to
            scores not seen during fitting.
          </p>
          <p>
            This is the usual choice for a categorisation that becomes part of an analysis or
            processing pipeline.
          </p>

          <h3>Optimize a fixed sample</h3>
          <p>Sometimes the rows you already have are the final object.</p>
          <pre className="home-article__code">
            <code>{"partition = sq.optimize_partition(...)\nlabels = partition.labels"}</code>
          </pre>
          <p>
            Here ScoreQuant can optimize their labels directly. The result may achieve a better
            finite-sample objective, but it does not by itself define how a new observation should
            be labelled.
          </p>
        </section>

        <section aria-labelledby="scores">
          <h2 id="scores">Where do the scores come from?</h2>
          <p>ScoreQuant starts from score vectors. They may come from:</p>
          <ul className="home-article__list">
            <li>an analytic or differentiable likelihood;</li>
            <li>scores computed elsewhere and supplied as an array;</li>
            <li>estimated density ratios or classifier-based surrogates.</li>
          </ul>
          <p>
            The optimization machinery is the same. What changes is the interpretation: exact model
            scores give Fisher information directly, while estimated scores give information
            relative to the learned surrogate.
          </p>
        </section>

        <section aria-labelledby="criterion">
          <h2 id="criterion">What is being optimized?</h2>
          <p>
            The default criterion is <strong>D-optimality</strong>: preserve information jointly
            across all parameter directions.
          </p>
          <p>
            Other criteria support problems such as nuisance parameters or different notions of
            information retention.
          </p>
          <p>
            The important distinction is that ScoreQuant optimizes the information relevant to
            estimation — not geometric compactness in the original observation space.
          </p>
        </section>

        <section aria-labelledby="next">
          <h2 id="next">Where next?</h2>

          <h3>
            <Link to="/get-started">Get started →</Link>
          </h3>
          <p>Fit a quantizer, inspect its D-efficiency, and use it on new scores.</p>

          <h3>
            <Link to="/walkthroughs">Walkthroughs →</Link>
          </h3>
          <p>
            See complete estimation problems, from the model and scores to the resulting hard
            categories.
          </p>

          <h3>
            <Link to="/research">Research →</Link>
          </h3>
          <p>
            Explore an atlas of the mathematics, guarantees, limitations, exact results, and open
            questions behind the method, one page per result.
          </p>
        </section>
      </article>
    </Layout>
  );
}
