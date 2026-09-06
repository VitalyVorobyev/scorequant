import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";

import {ReferenceLink} from "../components/ReferenceLink";
import {Tex} from "../components/Tex";

/**
 * The portal's front page: the definitions a reader needs before any other
 * page makes sense, and where each of them is derived. It quotes no
 * measurement (`tests/test_walkthrough_facts.py` holds it to that) and runs
 * nothing.
 */
export default function Home(): React.JSX.Element {
  return (
    <Layout
      title="ScoreQuant"
      description="Definitions: score, Fisher information, hard binning and what it costs, the two tasks, the criteria, and where each is derived."
    >
      <article className="home-article">
        <header>
          <h1>ScoreQuant</h1>
          <p className="home-article__lead">
            A Python library for information-preserving hard binning: it assigns observations to
            a fixed number of labels while retaining the Fisher information that parameter
            estimation depends on, and it reports what the binning cost.
          </p>
          <p className="home-article__install">
            <code>uv add scorequant</code>
          </p>
        </header>

        <section aria-labelledby="setting">
          <h2 id="setting">The setting</h2>
          <p>
            An observation <Tex>x</Tex> is drawn from a parametric model{" "}
            <Tex>{String.raw`p(x \mid \theta)`}</Tex>. A <em>reference point</em>{" "}
            <Tex>{String.raw`\theta_0`}</Tex> is fixed in advance: the current best estimate, the
            nominal calibration, the null hypothesis. Everything below is evaluated there.
          </p>
          <p>
            The <em>score</em> of an observation is the gradient of its log-likelihood at the
            reference point,
          </p>
          <Tex display>{String.raw`s(x) = \nabla_\theta \log p(x \mid \theta)\,\big|_{\theta_0},`}</Tex>
          <p>
            one coordinate per parameter, whatever the dimension of <Tex>x</Tex>. The{" "}
            <em>Fisher information</em> of one observation is the second moment of its score,{" "}
            <Tex>{String.raw`I = \mathbb{E}\big[s(X)\,s(X)^{\top}\big]`}</Tex>. For independent observations in a regular model, the inverse of the total information
            bounds the covariance of unbiased estimators of <Tex>{String.raw`\theta`}</Tex>{" "}
            when that information is nonsingular.
          </p>
        </section>

        <section aria-labelledby="binning">
          <h2 id="binning">Hard binning, and what it costs</h2>
          <p>
            A <em>hard binning</em> is a map <Tex>{String.raw`q : x \mapsto \{1, \dots, K\}`}</Tex>.
            For a fixed rule in a regular model, the bin label has per-observation Fisher
            information <Tex>{String.raw`I_q`}</Tex>. The difference is an identity, not an
            estimate:
          </p>
          <Tex display>{String.raw`I - I_q \;=\; \sum_{b=1}^{K} \mathbb{E}\Big[\,\mathbb{1}\{q(X)=b\}\,\big(s(X)-\mu_b\big)\big(s(X)-\mu_b\big)^{\top}\Big] \;\succeq\; 0,`}</Tex>
          <p>
            where <Tex>{String.raw`\mu_b`}</Tex> is the mean score of cell <Tex>b</Tex>. Two things
            follow. Binning never creates information. And the loss is the within-cell scatter of
            the <em>score</em>, not of the observation, so the space in which to choose cells is
            score space. ScoreQuant reports the <em>retention</em>, a normalised ratio of{" "}
            <Tex>{String.raw`I_q`}</Tex> to <Tex>I</Tex> according to the chosen criterion. Zero D-efficiency can mean that just one parameter
            direction was lost; it does not imply that no information remains.
          </p>
        </section>

        <section aria-labelledby="tasks">
          <h2 id="tasks">The task, stated twice</h2>
          <p>
            <strong>Partition a fixed table.</strong> Given <Tex>N</Tex> scores with weights,
            choose a label for each of those rows. <code>optimize_partition</code> returns a{" "}
            <code>PartitionResult</code>: the labels, the retained information and its
            certificates, and no predictor. A finite sample optimum does not name a rule for
            observations outside the sample. This is the task when the rows are the final object:
            a frozen template set, a calibration sample, a study of what a cell budget can retain.
          </p>
          <p>
            <strong>Fit a reusable rule.</strong> Given a source measure over observations and a
            score provider that converts observations to scores, choose a rule on score space
            that labels scores not yet seen. <code>fit_quantizer</code> returns a{" "}
            <code>QuantizerResult</code> whose <code>predict_scores</code> applies the rule
            anywhere. This is the task when future observations must be labelled the same way: a
            trigger, a gate, a categorisation shipped with an analysis.
          </p>
          <p>
            A full-D partition can be compiled after its stability and geometry are verified,
            reproducing positive-weight training labels. The library does not provide a generic
            compiler for profiled Ds.
          </p>
        </section>

        <section aria-labelledby="criteria">
          <h2 id="criteria">The criteria</h2>
          <p>
            A criterion turns the retained information matrix into one number to maximise.{" "}
            <strong>D-optimality</strong> maximises <Tex>{String.raw`\det I_q`}</Tex> over the
            informative subspace, weighing every parameter direction at once.{" "}
            <strong>Profiled <Tex>{String.raw`D_s`}</Tex>-optimality</strong> maximises the
            determinant of the Schur complement for the parameters of interest, after the nuisance
            parameters are profiled out. The <strong>normalised trace</strong> maximises{" "}
            <Tex>{String.raw`\operatorname{tr}(I^{-1} I_q)`}</Tex>, which after whitening is the
            <Tex>k</Tex>-means objective. Each criterion pairs with a fixed set of solvers, and an
            unsupported pair fails before anything runs; the{" "}
            <ReferenceLink to="method/">method overview</ReferenceLink> lists the pairs and{" "}
            <ReferenceLink to="user-workflow/">Choosing your workflow</ReferenceLink> decides
            between them.
          </p>
        </section>

        <section aria-labelledby="provenance">
          <h2 id="provenance">Where the scores come from</h2>
          <p>
            An <em>exact</em> score is the derivative of a model you can write down, or an array
            of scores you computed from one. An <em>estimated</em> score comes from density ratios,
            typically a calibrated classifier whose posterior odds over its prior odds estimate
            the likelihood ratio. The optimisation is the same in both cases. The information is
            not: an estimated score yields a <em>surrogate</em> information whose Fisher meaning
            is only as good as the estimate, and every result records which of the two it
            reported. <ReferenceLink to="three-doors/">Three doors</ReferenceLink> describes the
            input routes.
          </p>
        </section>

        <section aria-labelledby="reading">
          <h2 id="reading">Where each of these is derived</h2>
          <ul className="home-article__refs">
            <li>
              <ReferenceLink to="book/ch01-why-bin/">Why bin at all</ReferenceLink>: the score,
              the information, and the cost of binning on a model computable by hand.
            </li>
            <li>
              <ReferenceLink to="book/ch04-scores-and-doors/">Scores, score laws, and the three doors</ReferenceLink>:
              exact densities, density ratios, and precomputed scores as inputs.
            </li>
            <li>
              <ReferenceLink to="book/ch05-information-after-binning/">Information after hard labels</ReferenceLink>:
              the identity above, derived.
            </li>
            <li>
              <ReferenceLink to="book/ch06-two-tasks/">Two tasks and three optimisation levels</ReferenceLink>:
              why a partition has no predict method.
            </li>
            <li>
              <ReferenceLink to="book/ch08-d-optimality/">D-optimality and exact exchange</ReferenceLink>:
              the finite solver and the compile bridge.
            </li>
            <li>
              <ReferenceLink to="book/ch10-profiled-ds/">Nuisance parameters and profiled <Tex>{String.raw`D_s`}</Tex></ReferenceLink>:
              the Schur complement, the efficient-score ceiling, and what cannot be compiled.
            </li>
            <li>
              <ReferenceLink to="book/ch13-estimated-scores/">Estimated density ratios and scores</ReferenceLink>:
              what a surrogate information does and does not mean.
            </li>
            <li>
              <ReferenceLink to="api/">API guide</ReferenceLink> and the{" "}
              <ReferenceLink to="symbols/">generated reference</ReferenceLink> for every public
              object, including the refusals.
            </li>
            <li>
              <ReferenceLink to="examples/">Runnable examples</ReferenceLink>, each with its
              committed evidence file.
            </li>
            <li>
              <Link to="/get-started">Get started</Link>: one problem from installation to a
              rule, with the printed output of every step.
            </li>
            <li>
              <Link to="/walkthroughs">Walkthroughs</Link>: four applied problems, each on
              real or generated data with every number traced to a committed run.
            </li>
            <li>
              <Link to="/research">Research</Link>: an atlas of what is known, what the library
              established, where it fails, and what is open, one page per result.
            </li>
          </ul>
        </section>
      </article>
    </Layout>
  );
}
