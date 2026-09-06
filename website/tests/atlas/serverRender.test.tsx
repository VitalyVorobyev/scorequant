import {renderToStaticMarkup} from "react-dom/server";
import {describe, expect, it} from "vitest";

import AuthorPage from "../../src/atlas/pages/AuthorPage";
import FixturePage from "../../src/atlas/pages/FixturePage";
import LiteraturePage from "../../src/atlas/pages/LiteraturePage";
import PaperPage from "../../src/atlas/pages/PaperPage";
import {atlas, core, fixtureData, literatureData, renderedPaper} from "./renderData";

describe("server render", () => {
  it("renders without a window", () => {
    const lit = renderToStaticMarkup(<LiteraturePage core={core} data={literatureData()} />);
    expect(lit).toContain("year-strip");
    for (const fixture of Object.values(atlas.fixtures)) {
      expect(renderToStaticMarkup(<FixturePage core={core} data={fixtureData(fixture.id)} />)).toContain("Counterexample");
    }
    for (const key of Object.keys(atlas.papers)) renderToStaticMarkup(<PaperPage core={core} data={renderedPaper(key)} />);
    for (const author of Object.values(atlas.authors)) renderToStaticMarkup(<AuthorPage core={core} data={author} />);
  });
});
