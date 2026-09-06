import rawData from "../generated/portal-data.json";

export interface ApiSymbol {
  kind: "class" | "function";
  name: string;
  reference: string;
  signature: string;
  source: string;
  summary: string;
}

export interface PortalData {
  api: ApiSymbol[];
  schemaVersion: number;
}

export const portalData = rawData as PortalData;
