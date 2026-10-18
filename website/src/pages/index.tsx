import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";

import {Figure} from "../components/Figure";
import {ReferenceLink} from "../components/ReferenceLink";
import {Tex} from "../components/Tex";
import {WALKTHROUGHS} from "../data/walkthroughs";
import {siteUrl} from "../lib/site";

/**
 * The portal's front page: what the library does and one picture of it, then
 * why score space, what each walkthrough found, and where to start. It quotes
 * no measurement (`tests/test_walkthrough_facts.py` holds it to that) and runs
 * nothing. The two tasks are explained once, on the get-started page.
 */
export default function Home(): React.JSX.Element {
  return (
    <Layout
      title="ScoreQuant"
      description="ScoreQuant sorts events into a few hard bins chosen in score space, so the bins keep as much Fisher information about your parameters as possible."
    >
      <article className="home-article">
        <header>
          <h1>ScoreQuant</h1>
          <p className="home-article__lead">
            ScoreQuant sorts events into a few hard bins, chosen so that the bin counts keep as much
            information about your model parameters as possible, and it reports how much they kept.
          </p>
          <p className="home-article__install">
            <code>uv add scorequant</code> <Link to="/get-started">Get started →</Link>
          </p>
          <Figure
            src={siteUrl("walkthrough-figures/michelson-d-geometry.png")}
            alt="Three panels. Top left: the raw score plane tinted by six convex cells with straight edges, the trajectory coloured by cell. Top right: the same cells after whitening, an ordinary Voronoi diagram of the cell means. Bottom: the same labels along the detector, a comb of many intervals above six equal segments."
            caption={
              <>
                Bins for an interferometer, from the{" "}
                <Link to="/walkthroughs/michelson">Michelson walkthrough</Link>. In score space
                (left) each bin is a convex cell. Along the detector (bottom) the same bins form a
                comb, very unlike equal segments.
              </>
            }
            paper
          />
        </header>

        <section aria-labelledby="why">
          <h2 id="why">Why score space</h2>
          <p>
            Near a reference point, all an estimate uses from an event is its score{" "}
            <Tex>{String.raw`s(x)=\nabla_\theta \log p(x\mid\theta)`}</Tex>, one number per
            parameter. A bin keeps only its total weight and the mean score of its events, so the
            information it loses is the spread of scores inside it. Events with similar scores
            should therefore share a bin, however far apart they are in the original observation,
            and neighbours with different scores should not. ScoreQuant draws the bins in score
            space to keep information in every parameter direction at once (D-optimality).
          </p>
        </section>

        <section aria-labelledby="walkthroughs">
          <h2 id="walkthroughs">Walkthroughs</h2>
          <ul className="home-cards">
            {WALKTHROUGHS.map((card) => (
              <li key={card.slug} className="home-card">
                <Link to={card.href} className="home-card__title">
                  {card.title}
                </Link>
                <p>{card.result}</p>
              </li>
            ))}
          </ul>
        </section>

        <section aria-labelledby="start">
          <h2 id="start">Get started</h2>
          <p>
            <Link to="/get-started">Get started</Link> fits a reusable rule on a small example,
            applies it to new data and saves it. The{" "}
            <ReferenceLink to="guide/">user guide</ReferenceLink> covers inputs, criteria, solvers
            and nuisance parameters. The <Link to="/research">research page</Link> states the
            theorem behind the rule and its limits.
          </p>
        </section>
      </article>
    </Layout>
  );
}
