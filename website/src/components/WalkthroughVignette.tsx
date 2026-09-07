/**
 * One stroke-only vignette per walkthrough, drawn from the shape of the
 * problem rather than from its data: a fringe with unequal cells, two
 * overlapping densities with cut points, a histogram beside its shifted
 * copy, and a scatter cut by three cells. Colours come from the tokens, so
 * the figures follow the theme; nothing here is a measurement, so the
 * walkthrough fact contract does not apply.
 */
export type WalkthroughVignetteSlug = "flowcyt" | "hep" | "michelson" | "ratios";

const BODIES: Record<WalkthroughVignetteSlug, React.JSX.Element> = {
  michelson: (
    <>
      <line x1="0" y1="96" x2="150" y2="96" stroke="var(--border)" strokeWidth="1" />
      <line x1="14" y1="14" x2="14" y2="96" stroke="var(--text-muted)" strokeWidth="1" strokeDasharray="2 3" />
      <line x1="31" y1="14" x2="31" y2="96" stroke="var(--text-muted)" strokeWidth="1" strokeDasharray="2 3" />
      <line x1="44" y1="14" x2="44" y2="96" stroke="var(--text-muted)" strokeWidth="1" strokeDasharray="2 3" />
      <line x1="61" y1="14" x2="61" y2="96" stroke="var(--text-muted)" strokeWidth="1" strokeDasharray="2 3" />
      <line x1="84" y1="14" x2="84" y2="96" stroke="var(--text-muted)" strokeWidth="1" strokeDasharray="2 3" />
      <line x1="106" y1="14" x2="106" y2="96" stroke="var(--text-muted)" strokeWidth="1" strokeDasharray="2 3" />
      <line x1="124" y1="14" x2="124" y2="96" stroke="var(--text-muted)" strokeWidth="1" strokeDasharray="2 3" />
      <line x1="139" y1="14" x2="139" y2="96" stroke="var(--text-muted)" strokeWidth="1" strokeDasharray="2 3" />
      <polyline points="0,27.0 2,27.4 4,28.6 6,30.5 8,33.1 10,36.3 12,40.0 14,44.1 16,48.6 18,53.2 20,57.9 22,62.5 24,66.9 26,71.0 28,74.6 30,77.7 32,80.1 34,81.8 36,82.8 38,83.0 40,82.4 42,81.0 44,79.0 46,76.2 48,72.8 50,69.0 52,64.8 54,60.2 56,55.6 58,50.9 60,46.3 62,42.0 64,38.1 66,34.6 68,31.7 70,29.4 72,27.9 74,27.1 76,27.1 78,27.9 80,29.4 82,31.7 84,34.6 86,38.1 88,42.0 90,46.3 92,50.9 94,55.6 96,60.2 98,64.8 100,69.0 102,72.8 104,76.2 106,79.0 108,81.0 110,82.4 112,83.0 114,82.8 116,81.8 118,80.1 120,77.7 122,74.6 124,71.0 126,66.9 128,62.5 130,57.9 132,53.2 134,48.6 136,44.1 138,40.0 140,36.3 142,33.1 144,30.5 146,28.6 148,27.4 150,27.0" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinejoin="round" />
    </>
  ),
  ratios: (
    <>
      <line x1="0" y1="96" x2="150" y2="96" stroke="var(--border)" strokeWidth="1" />
      <line x1="52" y1="14" x2="52" y2="96" stroke="var(--text-muted)" strokeWidth="1" strokeDasharray="2 3" />
      <line x1="78" y1="14" x2="78" y2="96" stroke="var(--text-muted)" strokeWidth="1" strokeDasharray="2 3" />
      <line x1="104" y1="14" x2="104" y2="96" stroke="var(--text-muted)" strokeWidth="1" strokeDasharray="2 3" />
      <polyline points="0,85.2 2,83.7 4,82.1 6,80.4 8,78.5 10,76.5 12,74.4 14,72.1 16,69.7 18,67.2 20,64.6 22,61.9 24,59.2 26,56.4 28,53.5 30,50.7 32,47.9 34,45.2 36,42.5 38,39.9 40,37.5 42,35.3 44,33.2 46,31.4 48,29.8 50,28.4 52,27.4 54,26.6 56,26.2 58,26.0 60,26.2 62,26.6 64,27.4 66,28.4 68,29.8 70,31.4 72,33.2 74,35.3 76,37.5 78,39.9 80,42.5 82,45.2 84,47.9 86,50.7 88,53.5 90,56.4 92,59.2 94,61.9 96,64.6 98,67.2 100,69.7 102,72.1 104,74.4 106,76.5 108,78.5 110,80.4 112,82.1 114,83.7 116,85.2 118,86.5 120,87.7 122,88.8 124,89.8 126,90.6 128,91.4 130,92.1 132,92.7 134,93.2 136,93.6 138,94.0 140,94.3 142,94.6 144,94.9 146,95.1 148,95.2 150,95.4" fill="none" stroke="var(--border-strong)" strokeWidth="2" />
      <polyline points="0,96.0 2,96.0 4,96.0 6,96.0 8,96.0 10,96.0 12,96.0 14,96.0 16,96.0 18,96.0 20,96.0 22,96.0 24,96.0 26,96.0 28,96.0 30,96.0 32,96.0 34,96.0 36,96.0 38,96.0 40,96.0 42,96.0 44,95.9 46,95.9 48,95.9 50,95.8 52,95.7 54,95.5 56,95.3 58,95.0 60,94.6 62,94.1 64,93.4 66,92.4 68,91.2 70,89.7 72,87.9 74,85.7 76,83.0 78,80.0 80,76.5 82,72.7 84,68.5 86,64.1 88,59.6 90,55.1 92,50.7 94,46.6 96,43.1 98,40.1 100,37.8 102,36.5 104,36.0 106,36.5 108,37.8 110,40.1 112,43.1 114,46.6 116,50.7 118,55.1 120,59.6 122,64.1 124,68.5 126,72.7 128,76.5 130,80.0 132,83.0 134,85.7 136,87.9 138,89.7 140,91.2 142,92.4 144,93.4 146,94.1 148,94.6 150,95.0" fill="none" stroke="var(--accent)" strokeWidth="2" />
    </>
  ),
  hep: (
    <>
      <line x1="0" y1="96" x2="150" y2="96" stroke="var(--border)" strokeWidth="1" />
      <rect x="10" y="78" width="14" height="18" fill="none" stroke="var(--border-strong)" strokeWidth="1.5" />
      <rect x="27" y="62" width="14" height="34" fill="none" stroke="var(--border-strong)" strokeWidth="1.5" />
      <rect x="44" y="44" width="14" height="52" fill="none" stroke="var(--border-strong)" strokeWidth="1.5" />
      <rect x="61" y="26" width="14" height="70" fill="none" stroke="var(--border-strong)" strokeWidth="1.5" />
      <rect x="78" y="38" width="14" height="58" fill="none" stroke="var(--border-strong)" strokeWidth="1.5" />
      <rect x="95" y="56" width="14" height="40" fill="none" stroke="var(--border-strong)" strokeWidth="1.5" />
      <rect x="112" y="70" width="14" height="26" fill="none" stroke="var(--border-strong)" strokeWidth="1.5" />
      <rect x="129" y="80" width="14" height="16" fill="none" stroke="var(--border-strong)" strokeWidth="1.5" />
      <polyline points="16,77.8 33,59.9 50,39.8 67,19.6 84,33.0 101,53.2 118,68.9 135,80.1" fill="none" stroke="var(--accent)" strokeWidth="2" strokeDasharray="4 3" strokeLinejoin="round" />
    </>
  ),
  flowcyt: (
    <>
      <rect x="0.5" y="0.5" width="149" height="109" fill="none" stroke="var(--border)" strokeWidth="1" />
      <circle cx="36.4" cy="45.7" r="2.2" fill="var(--text-muted)" />
      <circle cx="36.8" cy="36.5" r="2.2" fill="var(--text-muted)" />
      <circle cx="27.0" cy="37.6" r="2.2" fill="var(--text-muted)" />
      <circle cx="55.6" cy="44.8" r="2.2" fill="var(--text-muted)" />
      <circle cx="54.5" cy="42.8" r="2.2" fill="var(--text-muted)" />
      <circle cx="45.5" cy="42.1" r="2.2" fill="var(--text-muted)" />
      <circle cx="16.7" cy="49.6" r="2.2" fill="var(--text-muted)" />
      <circle cx="47.1" cy="45.6" r="2.2" fill="var(--text-muted)" />
      <circle cx="16.3" cy="20.5" r="2.2" fill="var(--text-muted)" />
      <circle cx="27.5" cy="34.8" r="2.2" fill="var(--text-muted)" />
      <circle cx="44.3" cy="39.5" r="2.2" fill="var(--text-muted)" />
      <circle cx="106.3" cy="27.8" r="2.2" fill="var(--text-muted)" />
      <circle cx="103.7" cy="37.8" r="2.2" fill="var(--text-muted)" />
      <circle cx="92.1" cy="50.5" r="2.2" fill="var(--text-muted)" />
      <circle cx="106.7" cy="45.5" r="2.2" fill="var(--text-muted)" />
      <circle cx="92.6" cy="26.9" r="2.2" fill="var(--text-muted)" />
      <circle cx="95.9" cy="33.0" r="2.2" fill="var(--text-muted)" />
      <circle cx="107.6" cy="36.4" r="2.2" fill="var(--text-muted)" />
      <circle cx="94.6" cy="24.8" r="2.2" fill="var(--text-muted)" />
      <circle cx="93.8" cy="45.7" r="2.2" fill="var(--text-muted)" />
      <circle cx="90.3" cy="36.3" r="2.2" fill="var(--text-muted)" />
      <circle cx="105.1" cy="19.7" r="2.2" fill="var(--text-muted)" />
      <circle cx="80.6" cy="91.6" r="2.2" fill="var(--text-muted)" />
      <circle cx="53.8" cy="74.7" r="2.2" fill="var(--text-muted)" />
      <circle cx="78.6" cy="69.5" r="2.2" fill="var(--text-muted)" />
      <circle cx="86.5" cy="77.4" r="2.2" fill="var(--text-muted)" />
      <circle cx="61.0" cy="86.6" r="2.2" fill="var(--text-muted)" />
      <circle cx="88.7" cy="87.8" r="2.2" fill="var(--text-muted)" />
      <circle cx="98.7" cy="81.8" r="2.2" fill="var(--text-muted)" />
      <circle cx="81.6" cy="64.5" r="2.2" fill="var(--text-muted)" />
      <circle cx="88.0" cy="71.6" r="2.2" fill="var(--text-muted)" />
      <circle cx="74.1" cy="64.8" r="2.2" fill="var(--text-muted)" />
      <circle cx="67.4" cy="72.5" r="2.2" fill="var(--text-muted)" />
      <polyline points="70,8 66,58 22,86" fill="none" stroke="var(--accent)" strokeWidth="1.5" />
      <polyline points="66,58 150,64" fill="none" stroke="var(--accent)" strokeWidth="1.5" />
    </>
  ),
};

export function WalkthroughVignette({slug}: {slug: WalkthroughVignetteSlug}): React.JSX.Element {
  return (
    <svg className="walkthrough-card__vignette" viewBox="0 0 150 110" width={150} height={110} aria-hidden="true" focusable="false">
      {BODIES[slug]}
    </svg>
  );
}
