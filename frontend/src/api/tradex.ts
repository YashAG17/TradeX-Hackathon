import { apiPost } from "./client";
import type { TradexCompareResponse, TradexEpisodeResponse } from "./types";

export interface TradexEpisodePayload {
  seed: number;
  stage: number;
  use_overseer: boolean;
}

export interface TradexComparePayload {
  num_episodes: number;
}

export const tradexApi = {
  runEpisode: (payload: TradexEpisodePayload) =>
    apiPost<TradexEpisodeResponse>("/tradex/run-episode", payload),

  compare: (payload: TradexComparePayload) =>
    apiPost<TradexCompareResponse>("/tradex/compare", payload),
};
