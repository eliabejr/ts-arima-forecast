export interface ARIMAParams {
  p: number; // AR order
  d: number; // Differencing order
  q: number; // MA order
}

export interface SARIMAParams extends ARIMAParams {
  P: number; // Seasonal AR order
  D: number; // Seasonal differencing order
  Q: number; // Seasonal MA order
  s: number; // Seasonal period
}

export interface SARIMAXParams extends SARIMAParams {
  exogenous?: number[][]; // External variables
}

export interface ForecastResult {
  forecast: number[];
  lowerBound: number[];
  upperBound: number[];
  residuals: number[];
  aic: number;
  bic: number;
  logLikelihood: number;
}

export interface ModelFitResult {
  coefficients: {
    ar?: number[];
    ma?: number[];
    sar?: number[];
    sma?: number[];
    exog?: number[];
  };
  residuals: number[];
  fittedValues: number[];
  aic: number;
  bic: number;
  logLikelihood: number;
  sigma2: number;
}