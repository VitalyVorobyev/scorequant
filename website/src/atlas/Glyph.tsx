import type {Provenance} from "../data/atlas";

export type GlyphClass = Provenance | "fixture";

export interface GlyphProps {
  /** The provenance class, or `fixture` for a counterexample fixture. */
  kind: GlyphClass;
  /** Draw the machine-checked mark. */
  checked?: boolean;
  /** Accessible name; when omitted the glyph is decorative. */
  label?: string;
  size?: "md" | "lg";
}

/**
 * The one glyph system of the atlas. Shape and fill together encode the
 * provenance class, so the classes stay distinguishable without colour:
 * established and derived are filled discs in two inks; proved here is the
 * accent disc; proved here with no precedent found adds an outer ring;
 * measured is a hatched disc; a counterexample is a square with a cut
 * corner; an open question is a hollow disc; a verification record is a
 * hollow diamond; a fixture is a small triangle. A tick badge marks a
 * machine-checked statement. Rendered inline everywhere a claim is named,
 * and as the node symbol of every map.
 */
export function Glyph({kind, checked = false, label, size = "md"}: GlyphProps): React.JSX.Element {
  const className = `glyph glyph-${kind}${size === "lg" ? " glyph--lg" : ""}`;
  return (
    <svg className={className} viewBox="0 0 16 16" role={label ? "img" : undefined} aria-hidden={label ? undefined : true}>
      {label ? <title>{label}</title> : null}
      <GlyphShape kind={kind} />
      {checked ? <CheckBadge /> : null}
    </svg>
  );
}

/** The bare shape, for use inside a larger SVG (maps). Coordinates span 0–16. */
export function GlyphShape({kind}: {kind: GlyphClass}): React.JSX.Element {
  switch (kind) {
    case "proved_new":
      return (
        <>
          <circle className="glyph__ring" cx="8" cy="8" r="7" strokeWidth="1.3" />
          <circle cx="8" cy="8" r="4.2" />
        </>
      );
    case "measured":
      return (
        <>
          <circle cx="8" cy="8" r="6" />
          <g className="glyph__hatch">
            <path d="M3 10 L10 3 M5 13 L13 5 M2 6 L6 2 M8 15 L15 8" />
          </g>
        </>
      );
    case "counterexample":
      return <path d="M2.5 2.5 H10.5 L13.5 5.5 V13.5 H2.5 Z" strokeWidth="1" />;
    case "open":
      return <circle cx="8" cy="8" r="5.6" strokeWidth="1.7" />;
    case "verification":
      return <path d="M8 2 L14 8 L8 14 L2 8 Z" strokeWidth="1.4" />;
    case "fixture":
      return <path d="M8 2.5 L14 13.5 H2 Z" strokeWidth="1" />;
    case "established":
    case "derived":
    case "proved":
    default:
      return <circle cx="8" cy="8" r="6" />;
  }
}

function CheckBadge(): React.JSX.Element {
  return (
    <g className="glyph__check" transform="translate(8.5 8.5)">
      <circle cx="4" cy="4" r="4" />
      <path d="M2 4.2 L3.5 5.6 L6.2 2.6" />
    </g>
  );
}
