import Statistics from '../utils/statistics';
import Preprocessing from '../utils/preprocessing';
import { ARIMAParams, ForecastResult, ModelFitResult } from '../types';

export class ARIMA {
  private params: ARIMAParams;
  private data: number[];
  private fitResult?: ModelFitResult;

  constructor(params: ARIMAParams) {
    this.params = params;
    this.data = [];
  }

  fit(data: number[]): ModelFitResult {
    this.data = [...data];

    let workingData = [...data];
    for (let i = 0; i < this.params.d; i++) {
      workingData = Preprocessing.difference(workingData);
    }

    const { p, q } = this.params;
    const n = workingData.length;

    const arParams = this.estimateARParameters(workingData, p);
    const maParams = this.estimateMAParameters(workingData, arParams, q);

    const fittedValues = this.calculateFittedValues(workingData, arParams, maParams);
    const residuals = workingData.slice(Math.max(p, q)).map((val, i) => val - fittedValues[i]);

    const sigma2 = Statistics.variance(residuals);
    const logLikelihood = this.calculateLogLikelihood(residuals, sigma2);
    const numParams = p + q + 1;
    const aic = Statistics.akaike(logLikelihood, numParams, n);
    const bic = Statistics.bayesian(logLikelihood, numParams, n);

    this.fitResult = {
      coefficients: {
        ar: arParams,
        ma: maParams
      },
      residuals,
      fittedValues,
      aic,
      bic,
      logLikelihood,
      sigma2
    };

    return this.fitResult;
  }

  private estimateARParameters(data: number[], p: number): number[] {
    if (p === 0) return [];

    const n = data.length;
    const X: number[][] = [];
    const y: number[] = [];

    for (let i = p; i < n; i++) {
      const row: number[] = [];
      for (let j = 1; j <= p; j++) {
        row.push(data[i - j]);
      }
      X.push(row);
      y.push(data[i]);
    }

    // Solve using normal equations: (X'X)^-1 X'y
    return Statistics.leastSquares(X, y);
  }

  private estimateMAParameters(data: number[], arParams: number[], q: number): number[] {
    if (q === 0) return [];

    const maParams = new Array(q).fill(0.1);

    for (let iter = 0; iter < 10; iter++) {
      const residuals = this.calculateResiduals(data, arParams, maParams);

      for (let i = 0; i < q && i < residuals.length - 1; i++) {
        if (residuals.length > i + 1) {
          maParams[i] = -Statistics.autocorrelation(residuals, i + 1) * 0.8;
        }
      }
    }

    return maParams;
  }

  private calculateResiduals(data: number[], arParams: number[], maParams: number[]): number[] {
    const p = arParams.length;
    const q = maParams.length;
    const n = data.length;
    const residuals: number[] = [];

    for (let i = Math.max(p, q); i < n; i++) {
      let prediction = 0;

      for (let j = 0; j < p; j++) {
        prediction += arParams[j] * data[i - j - 1];
      }

      for (let j = 0; j < q && j < residuals.length; j++) {
        prediction += maParams[j] * residuals[residuals.length - j - 1];
      }

      residuals.push(data[i] - prediction);
    }

    return residuals;
  }

  private calculateFittedValues(data: number[], arParams: number[], maParams: number[]): number[] {
    const residuals = this.calculateResiduals(data, arParams, maParams);
    return data.slice(Math.max(arParams.length, maParams.length)).map((val, i) => val - residuals[i]);
  }

  private calculateLogLikelihood(residuals: number[], sigma2: number): number {
    const n = residuals.length;
    const sumSquares = residuals.reduce((sum, r) => sum + r * r, 0);
    return -n / 2 * Math.log(2 * Math.PI) - n / 2 * Math.log(sigma2) - sumSquares / (2 * sigma2);
  }

  forecast(steps: number, confidenceLevel: number = 0.95): ForecastResult {
    if (!this.fitResult) {
      throw new Error('Model must be fitted before forecasting');
    }

    const { ar: arParams = [], ma: maParams = [] } = this.fitResult.coefficients;
    const { residuals, sigma2 } = this.fitResult;

    const forecast: number[] = [];
    const variance: number[] = [];

    let workingData = [...this.data];
    for (let i = 0; i < this.params.d; i++) {
      workingData = Preprocessing.difference(workingData);
    }

    const lastResiduals = residuals.slice(-Math.max(maParams.length, 1));

    for (let h = 1; h <= steps; h++) {
      let prediction = 0;
      let predVariance = sigma2;

      for (let i = 0; i < arParams.length; i++) {
        if (h - i - 1 >= 0 && h - i - 1 < forecast.length) {
          prediction += arParams[i] * forecast[h - i - 2];
        } else if (workingData.length - i - 1 >= 0) {
          prediction += arParams[i] * workingData[workingData.length - i - 1];
        }
      }

      for (let i = 0; i < maParams.length && h <= i + 1; i++) {
        if (lastResiduals.length - (h - i - 1) - 1 >= 0) {
          prediction += maParams[i] * lastResiduals[lastResiduals.length - (h - i - 1) - 1];
        }
      }

      forecast.push(prediction);
      variance.push(predVariance);
    }

    let finalForecast = [...forecast];
    for (let i = 0; i < this.params.d; i++) {
      finalForecast = Preprocessing.undifference(
        this.data.slice(-(this.params.d - i)),
        finalForecast
      );
    }

    // calculate confidence intervals
    const zScore = Statistics.getZScore(confidenceLevel);
    const lowerBound = finalForecast.map((f, i) => f - zScore * Math.sqrt(variance[i]));
    const upperBound = finalForecast.map((f, i) => f + zScore * Math.sqrt(variance[i]));

    return {
      forecast: finalForecast,
      lowerBound,
      upperBound,
      residuals: this.fitResult.residuals,
      aic: this.fitResult.aic,
      bic: this.fitResult.bic,
      logLikelihood: this.fitResult.logLikelihood
    };
  }

  getParams(): ARIMAParams {
    return { ...this.params };
  }

  getFitResult(): ModelFitResult | undefined {
    return this.fitResult;
  }
}