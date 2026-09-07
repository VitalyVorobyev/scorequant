import Link from "@docusaurus/Link";

import {TAG_KIND_LABELS, WALKTHROUGHS, type TagKind, type WalkthroughCard} from "../data/walkthroughs";
import {WalkthroughVignette, type WalkthroughVignetteSlug} from "./WalkthroughVignette";

const TAG_ORDER: readonly TagKind[] = ["task", "input", "criterion", "solver"];

/** The two tasks, in the words the front page uses for them. */
const TASK_BADGES: Record<string, string> = {
  fit_quantizer: "Fits a reusable rule",
  optimize_partition: "Labels a fixed sample"
};

function Card({card}: {card: WalkthroughCard}): React.JSX.Element {
  const badges = card.tags.filter((tag) => tag.kind === "task").map((tag) => TASK_BADGES[tag.label] ?? tag.label);
  const symbols = TAG_ORDER.filter((kind) => kind !== "task").flatMap((kind) => card.tags.filter((tag) => tag.kind === kind));
  return (
    <article className="walkthrough-card" aria-labelledby={`walkthrough-${card.slug}`}>
      <WalkthroughVignette slug={card.slug as WalkthroughVignetteSlug} />
      <div className="walkthrough-card__body">
        <div className="walkthrough-card__badges">
          {badges.map((badge) => (
            <span key={badge} className="walkthrough-card__badge">
              {badge}
            </span>
          ))}
        </div>
        <h2 id={`walkthrough-${card.slug}`}>
          <Link to={card.href}>{card.title}</Link>
        </h2>
        <p className="walkthrough-card__lead">{card.lead}</p>
        <p className="walkthrough-card__problem">{card.problem}</p>
        <p className="walkthrough-card__data">
          <span>Data</span> {card.data}
        </p>
        <dl className="walkthrough-card__tags">
          {card.tags
            .filter((tag) => tag.kind === "task")
            .map((tag) => (
              <div key={tag.label} className="visually-hidden">
                <dt>{TAG_KIND_LABELS.task}</dt>
                <dd>{tag.label}</dd>
              </div>
            ))}
          {symbols.map((tag) => (
            <div key={tag.label}>
              <dt className="visually-hidden">{TAG_KIND_LABELS[tag.kind]}</dt>
              <dd>
                <code title={TAG_KIND_LABELS[tag.kind]}>{tag.label}</code>
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </article>
  );
}

/** The card grid on the walkthroughs index: four problems, comparable at a glance. */
export function WalkthroughCards(): React.JSX.Element {
  return (
    <div className="walkthrough-cards walkthrough-index">
      {WALKTHROUGHS.map((card) => (
        <Card key={card.slug} card={card} />
      ))}
    </div>
  );
}
