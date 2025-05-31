import { ARIMA } from '../models/arima';

export interface ModelSelection {
  bestParams: any;
  bestScore: number;
  allResults: Array<{ params: any; score: number }>;
}

export class AutoARIMA {
  static findBestARIMA(
    data: number[],
    maxP: number = 5,
    maxD: number = 2,
    maxQ: number = 5,
    criterion: 'aic' | 'bic' = 'aic'
  ): ModelSelection {
    const results: Array<{ params: any; score: number }> = [];
    let bestScore = Infinity;
    let bestParams: any = null;

    for (let p = 0; p <= maxP; p++) {
      for (let d = 0; d <= maxD; d++) {
        for (let q = 0; q <= maxQ; q++) {
          try {
            const model = new ARIMA({ p, d, q });
            const fitResult = model.fit(data);
            const score = criterion === 'aic' ? fitResult.aic : fitResult.bic;

            results.push({ params: { p, d, q }, score });

            if (score < bestScore) {
              bestScore = score;
              bestParams = { p, d, q };
            }
          } catch (error) {
            continue;
          }
        }
      }
    }

    return {
      bestParams,
      bestScore,
      allResults: results.sort((a, b) => a.score - b.score)
    };
  }
}