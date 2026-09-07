import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps, CoreClaim} from "../core";
import {ResultRow} from "../ResultRow";
import {useAtlasQuery} from "../useAtlasQuery";
import LandscapeMatrix from "./LandscapeMatrix";

const INTRO: Record<string, string> = {
  universal: "What any hard label retains, loses, and can represent.",
  screening: "When a first-order bound can safely rule out a finite move.",
  d: "The geometry, exact gains, and limits of determinant-optimal partitions.",
  ds: "What changes when nuisance parameters are profiled out.",
  controls: "Alternative criteria reveal which conclusions depend on the determinant.",
  soft: "Randomized labels, smooth optimization, and the return to hard bins.",
  consistency: "When a finite-sample solution says something about the population.",
  access: "From densities and classifiers to scores, and from proxy to true information.",
};
const BANDS = ["Known", "Established here", "Boundary", "Open"] as const;
function band(c: CoreClaim): (typeof BANDS)[number] {
  if (c.provenance === "open") return "Open";
  if (c.provenance === "counterexample") return "Boundary";
  if (c.provenance === "established" || c.provenance === "derived") return "Known";
  return "Established here";
}

/** Theme-first exploration with an optional full criterion matrix. */
export default function LandscapePage({core, data}: AtlasPageProps<Record<string, never>>): React.JSX.Element {
  const {params, update} = useAtlasQuery();
  const query = params.get("q") ?? "";
  const theme = params.get("theme") ?? "";
  const matrix = params.get("view") === "matrix";
  const criteria = params.get("criterion") ?? "";
  const level = params.get("level") ?? "";
  const provenance = params.get("provenance") ?? "";
  const hasFilter = !!(query || theme || criteria || level || provenance);
  const claims = Object.values(core.claims)
    .filter(
      (c) =>
        c.kind !== "audit" &&
        (!theme || c.strip === theme || (theme === "other" && !c.strip)) &&
        (!criteria || c.criterion.includes(criteria)) &&
        (!level || c.level === level) &&
        (!provenance || c.provenance === provenance) &&
        `${c.editorial?.title} ${c.editorial?.summary} ${c.title} ${c.id}`.toLowerCase().includes(query.toLowerCase()),
    )
    .sort((a, b) => (a.editorial?.title ?? a.title).localeCompare(b.editorial?.title ?? b.title));
  const themeList = (<div className="research-theme-list" aria-label="Research themes">
            {[...core.strips, {id: "other", label: "Further results"}]
              .filter((t) => t.id !== "other" || Object.values(core.claims).some((c) => c.kind !== "audit" && !c.strip))
              .map((t) => (
                <button key={t.id} type="button" aria-pressed={theme === t.id} onClick={() => update({theme: theme === t.id ? "" : t.id})}>
                  <strong>{t.label}</strong>
                  <span>{INTRO[t.id] ?? "Research recorded outside the main themes."}</span>
                </button>
              ))}
          </div>);
  return (
    <AtlasShell title="Explore" description="Explore known results, contributions, boundaries, and open questions by research theme." wide>
      <h1>Explore the research</h1>
      <p className="atlas__lead atlas__measure">
        Choose a theme to follow its results, or search across the record. Each entry leads to its exact statement, assumptions, and evidence.
      </p>
      <div className="research-view-switch">
        <button type="button" aria-pressed={!matrix} onClick={() => update({view: ""})}>
          By theme
        </button>
        <button type="button" aria-pressed={matrix} onClick={() => update({view: "matrix"})}>
          Advanced matrix
        </button>
      </div>
      {matrix ? (
        <LandscapeMatrix core={core} data={data} />
      ) : (
        <>
          <div className="research-search">
            <label htmlFor="explore-search">Search results</label>
            <input
              id="explore-search"
              type="search"
              value={query}
              placeholder="geometry, nuisance, estimated scores…"
              onChange={(e) => update({q: e.target.value}, true)}
            />
          </div>
          <details className="research-disclosure">
            <summary>Filter by criterion, problem level, or attribution</summary>
            <div className="research-filters">
              {(
                [
                  ["criterion", "Criterion", core.vocabulary.criteria],
                  ["level", "Problem level", core.vocabulary.levels],
                  ["provenance", "Attribution", core.vocabulary.provenance.filter((p) => p.id !== "verification")],
                ] as const
              ).map(([key, label, options]) => (
                <label key={key}>
                  {label}
                  <select value={params.get(key) ?? ""} onChange={(e) => update({[key]: e.target.value})}>
                    <option value="">All</option>
                    {options.map((o) => (
                      <option key={o.id} value={o.id}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
            </div>
          </details>
          {hasFilter ? <details className="research-disclosure"><summary>Change research theme</summary>{themeList}</details> : themeList}
          {hasFilter ? (
            <div className="research-results">
              <div className="research-results__heading">
                <h2>{core.strips.find((t) => t.id === theme)?.label ?? "Matching results"}</h2>
                <button type="button" onClick={() => update({q: "", theme: "", criterion: "", level: "", provenance: ""})}>
                  Clear filters
                </button>
              </div>
              <p role="status">
                {claims.length ? `${claims.length} results in this selection.` : "No results recorded for this selection. Try clearing a filter."}
              </p>
              {BANDS.map((name) => {
                const selected = claims.filter((c) => band(c) === name);
                return selected.length ? (
                  <section key={name} aria-label={name}>
                    <h2>{name}</h2>
                    {selected.map((c) => (
                      <ResultRow key={c.id} core={core} claim={c} />
                    ))}
                  </section>
                ) : null;
              })}
            </div>
          ) : (
            <p className="atlas__muted">
              Select a theme above to read its results. The advanced matrix compares criteria and problem levels across the record.
            </p>
          )}
        </>
      )}
    </AtlasShell>
  );
}
