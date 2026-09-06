import type {Atlas} from "../../src/data/atlas";
import type {Core, CoreClaim, CoreFixture, CorePaper} from "../../src/atlas/core";

export function coreClaim(claim: Atlas["claims"][string]): CoreClaim;
export function coreFixture(fixture: Atlas["fixtures"][string]): CoreFixture;
export function corePaper(paper: Atlas["papers"][string]): CorePaper;
export function buildCore(atlas: Atlas): Core;
