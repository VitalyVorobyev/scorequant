/**
 * The real atlas, as the pages see it, for component tests.
 *
 * `core` is exactly what the plugin passes every page; `atlas` is the full
 * generated document for building a route's `data` prop by hand. Rendered
 * HTML fields (`*Html`) are produced by the plugin at build time; tests that
 * need one pass a literal string.
 */
import {buildCore} from "../plugins/research-atlas/core.mjs";
import {atlas} from "../src/data/atlas";

export {atlas};
export const core = buildCore(atlas);
