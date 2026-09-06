import clsx from "clsx";

export interface FigureProps {
  alt: string;
  caption: React.ReactNode;
  /** Mat a light-background image on a permanently white card, see `.chart-figure--paper`. */
  paper?: boolean;
  src: string;
  wide?: boolean;
}

/**
 * A captioned static image, sharing the `.chart-figure` frame the SVG charts
 * already use so a page mixes generated plots and rendered images without a
 * visible seam.
 *
 * `src` is a site-rooted URL, so callers pass it through `siteUrl()`; the file
 * itself belongs in `website/static/figures/`, which is committed, and never in
 * `website/static/walkthrough-figures/`, which is generated and gitignored.
 */
export function Figure({alt, caption, paper = false, src, wide = false}: FigureProps): React.JSX.Element {
  return (
    <figure
      className={clsx("chart-figure", wide && "chart-figure--wide", paper && "chart-figure--paper")}
    >
      <img alt={alt} src={src} />
      <figcaption>{caption}</figcaption>
    </figure>
  );
}
