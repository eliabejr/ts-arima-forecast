import Statistics from './statistics';

export class Diagnostics {
  static ljungBox(residuals: number[], lags: number = 10): { statistic: number; pValue: number } {
    const n = residuals.length;
    const autocorrs: number[] = [];

    for (let lag = 1; lag <= lags; lag++) {
      autocorrs.push(Statistics.autocorrelation(residuals, lag));
    }

    let statistic = 0;
    for (let i = 0; i < lags; i++) {
      statistic += Math.pow(autocorrs[i], 2) / (n - i - 1);
    }
    statistic *= n * (n + 2);

    const pValue = this.chiSquarePValue(statistic, lags);

    return { statistic, pValue };
  }

  static jarqueBera(residuals: number[]): { statistic: number; pValue: number } {
    const n = residuals.length;
    const mean = Statistics.mean(residuals);
    const std = Statistics.standardDeviation(residuals);

    let skewness = 0;
    for (const r of residuals) {
      skewness += Math.pow((r - mean) / std, 3);
    }
    skewness /= n;

    let kurtosis = 0;
    for (const r of residuals) {
      kurtosis += Math.pow((r - mean) / std, 4);
    }
    kurtosis = kurtosis / n - 3;

    const statistic = n / 6 * (Math.pow(skewness, 2) + Math.pow(kurtosis, 2) / 4);
    const pValue = this.chiSquarePValue(statistic, 2);

    return { statistic, pValue };
  }

  private static chiSquarePValue(statistic: number, degreesOfFreedom: number): number {
    function gammaIncompleteUpper(s: number, x: number): number {
      let sum = 1, term = 1;
      for (let k = 1; k < 100; k++) {
        term *= x / (s + k);
        sum += term;
        if (term < 1e-10) break;
      }
      return Math.exp(-x + s * Math.log(x) - logGamma(s)) * sum;
    }

    function logGamma(z: number): number {
      const g = 7; // lanczos approximation
      const p = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059,
        12.507343278686905, -0.13857109526572012,
        9.9843695780195716e-6, 1.5056327351493116e-7
      ];
      if (z < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
      z -= 1;
      let x = p[0];
      for (let i = 1; i < g + 2; i++) x += p[i] / (z + i);
      const t = z + g + 0.5;
      return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x) - Math.log(z + 1);
    }

    const x = statistic / 2;
    const s = degreesOfFreedom / 2;
    return gammaIncompleteUpper(s, x);
  }


  static plotResiduals(residuals: number[]): {
    mean: number;
    variance: number;
    autocorrelations: number[];
    ljungBox: { statistic: number; pValue: number };
    jarqueBera: { statistic: number; pValue: number };
  } {
    const mean = Statistics.mean(residuals);
    const variance = Statistics.variance(residuals);
    const autocorrelations: number[] = [];

    for (let lag = 1; lag <= Math.min(20, Math.floor(residuals.length / 4)); lag++) {
      autocorrelations.push(Statistics.autocorrelation(residuals, lag));
    }

    return {
      mean,
      variance,
      autocorrelations,
      ljungBox: this.ljungBox(residuals),
      jarqueBera: this.jarqueBera(residuals)
    };
  }
}