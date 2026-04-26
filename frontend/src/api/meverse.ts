import { apiGet, apiPost, apiPostForm } from "./client";
import type {
  ComparePoliciesResponse,
  RunEpisodeResponse,
  TelemetryResponse,
} from "./types";

export interface RunEpisodePayload {
  task: string;
  policy: string;
  seed: number | null;
}

export interface ComparePoliciesPayload {
  task: string;
  seed: number;
}

export const meverseApi = {
  listTasks: () => apiGet<{ tasks: string[] }>("/meverse/tasks"),

  runEpisode: (payload: RunEpisodePayload) =>
    apiPost<RunEpisodeResponse>("/meverse/run-episode", payload),

  comparePolicies: (payload: ComparePoliciesPayload) =>
    apiPost<ComparePoliciesResponse>("/meverse/compare-policies", payload),

  uploadTelemetry: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return apiPostForm<TelemetryResponse>("/meverse/telemetry", fd);
  },
};
