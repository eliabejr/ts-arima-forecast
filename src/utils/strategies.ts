import { ARIMA } from "../models/arima";

export interface ForecastResult {
  forecasts: number[];
  confidenceIntervals: { lower: number[]; upper: number[] };
}

export interface StepwiseForecastResult extends ForecastResult {
  updatedModels: ARIMA[];
  errors: number[];
  actualValues: number[];
}

export interface RealTimeForecastResult extends ForecastResult {
  errors?: number[];
  actualValues?: number[];
}

export abstract class ForecastStrategy {
  protected baseModel: ARIMA;
  protected initialData: number[];

  constructor(baseModel: ARIMA, initialData: number[]) {
    this.baseModel = baseModel;
    this.initialData = [...initialData];
  }

  abstract forecast(steps: number): ForecastResult;
  abstract forecastWithRealTimeData(actualObservations: number[]): RealTimeForecastResult;
  abstract addObservationAndForecast(newObservation: number): { forecast: number; lowerBound: number; upperBound: number; error?: number };
}

export class Stepwise extends ForecastStrategy {
  private refitModel: boolean;
  private verbose: boolean;
  private currentData: number[];
  private currentModel: ARIMA;
  private lastForecast: { forecast: number; lowerBound: number; upperBound: number } | null = null;

  constructor(
    baseModel: ARIMA,
    initialData: number[],
    options: {
      refitModel?: boolean;
      verbose?: boolean;
    } = {}
  ) {
    super(baseModel, initialData);
    this.refitModel = options.refitModel ?? true;
    this.verbose = options.verbose ?? false;
    this.currentData = [...initialData];
    this.currentModel = baseModel;
  }

  forecast(steps: number): ForecastResult {
    const result = this.currentModel.forecast(steps);
    return {
      forecasts: result.forecast,
      confidenceIntervals: {
        lower: result.lowerBound,
        upper: result.upperBound
      }
    };
  }

  forecastWithRealTimeData(actualObservations: number[]): StepwiseForecastResult {
    const steps = actualObservations.length;
    const forecasts: number[] = [];
    const lowerBounds: number[] = [];
    const upperBounds: number[] = [];
    const updatedModels: ARIMA[] = [];
    const errors: number[] = [];

    let workingData = [...this.currentData];
    let workingModel = this.currentModel;

    for (let step = 0; step < steps; step++) {
      const oneStepForecast = workingModel.forecast(1);
      const forecastValue = oneStepForecast.forecast[0];
      const lowerBound = oneStepForecast.lowerBound[0];
      const upperBound = oneStepForecast.upperBound[0];

      forecasts.push(forecastValue);
      lowerBounds.push(lowerBound);
      upperBounds.push(upperBound);

      const actualObservation = actualObservations[step];
      const error = Math.abs(forecastValue - actualObservation);
      errors.push(error);

      workingData.push(actualObservation);

      // refit model if enabled
      if (this.refitModel) {
        const updatedModel = new ARIMA(workingModel.getParams());
        updatedModel.fit(workingData);
        workingModel = updatedModel;
        updatedModels.push(updatedModel);
      }

      if (this.verbose) {
        console.log(`Step ${step + 1}: Forecast=${forecastValue.toFixed(3)}, Actual=${actualObservation.toFixed(3)}, Error=${error.toFixed(3)}`);
      }
    }

    this.currentData = workingData;
    this.currentModel = workingModel;

    return {
      forecasts,
      confidenceIntervals: { lower: lowerBounds, upper: upperBounds },
      updatedModels,
      errors,
      actualValues: actualObservations
    };
  }

  addObservationAndForecast(newObservation: number): { forecast: number; lowerBound: number; upperBound: number; error?: number } {
    let error: number | undefined;

    if (this.lastForecast) {
      error = Math.abs(this.lastForecast.forecast - newObservation);

      if (this.verbose) {
        console.log(`Previous forecast: ${this.lastForecast.forecast.toFixed(3)}, Actual observation: ${newObservation.toFixed(3)}, Error: ${error.toFixed(3)}`);
      }
    }

    this.currentData.push(newObservation);

    if (this.refitModel) {
      const updatedModel = new ARIMA(this.currentModel.getParams());
      updatedModel.fit(this.currentData);
      this.currentModel = updatedModel;
    }

    const nextForecast = this.currentModel.forecast(1);
    const forecastResult = {
      forecast: nextForecast.forecast[0],
      lowerBound: nextForecast.lowerBound[0],
      upperBound: nextForecast.upperBound[0]
    };

    this.lastForecast = forecastResult;

    if (this.verbose) {
      console.log(`Next forecast: ${forecastResult.forecast.toFixed(3)}`);
    }

    return {
      ...forecastResult,
      error
    };
  }

  reset(newInitialData: number[]): void {
    this.currentData = [...newInitialData];
    this.currentModel = new ARIMA(this.baseModel.getParams());
    this.currentModel.fit(this.currentData);
    this.lastForecast = null;
  }

  getCurrentData(): number[] {
    return [...this.currentData];
  }

  getCurrentModel(): ARIMA {
    return this.currentModel;
  }
}

export class RollingWindow extends ForecastStrategy {
  private windowSize: number;
  private verbose: boolean;
  private currentData: number[];
  private lastForecast: { forecast: number; lowerBound: number; upperBound: number } | null = null;

  constructor(
    baseModel: ARIMA,
    initialData: number[],
    options: {
      windowSize?: number;
      verbose?: boolean;
    } = {}
  ) {
    super(baseModel, initialData);
    this.windowSize = options.windowSize ?? initialData.length;
    this.verbose = options.verbose ?? false;
    this.currentData = [...initialData];
  }

  forecast(steps: number): ForecastResult {
    // Use current window to forecast multiple steps ahead
    const windowData = this.currentData.slice(-this.windowSize);
    const model = new ARIMA(this.baseModel.getParams());
    model.fit(windowData);

    const result = model.forecast(steps);
    return {
      forecasts: result.forecast,
      confidenceIntervals: {
        lower: result.lowerBound,
        upper: result.upperBound
      }
    };
  }

  forecastWithRealTimeData(actualObservations: number[]): RealTimeForecastResult {
    const steps = actualObservations.length;
    const forecasts: number[] = [];
    const lowerBounds: number[] = [];
    const upperBounds: number[] = [];
    const errors: number[] = [];

    let workingData = [...this.currentData];

    for (let step = 0; step < steps; step++) {
      // Keep only the most recent observations (rolling window)
      const windowData = workingData.slice(-this.windowSize);

      // Fit model on current window
      const model = new ARIMA(this.baseModel.getParams());
      model.fit(windowData);

      // Forecast one step ahead
      const oneStepForecast = model.forecast(1);
      const forecastValue = oneStepForecast.forecast[0];
      const lowerBound = oneStepForecast.lowerBound[0];
      const upperBound = oneStepForecast.upperBound[0];

      // Store results
      forecasts.push(forecastValue);
      lowerBounds.push(lowerBound);
      upperBounds.push(upperBound);

      // Use actual observation and calculate error
      const actualObservation = actualObservations[step];
      const error = Math.abs(forecastValue - actualObservation);
      errors.push(error);

      workingData.push(actualObservation);

      if (this.verbose) {
        console.log(`Rolling Step ${step + 1}: Forecast=${forecastValue.toFixed(3)}, Actual=${actualObservation.toFixed(3)}, Error=${error.toFixed(3)}, Window Size=${windowData.length}`);
      }
    }

    // Update internal state
    this.currentData = workingData;

    return {
      forecasts,
      confidenceIntervals: { lower: lowerBounds, upper: upperBounds },
      errors,
      actualValues: actualObservations
    };
  }

  addObservationAndForecast(newObservation: number): { forecast: number; lowerBound: number; upperBound: number; error?: number } {
    let error: number | undefined;

    // Calculate error if we have a previous forecast
    if (this.lastForecast) {
      error = Math.abs(this.lastForecast.forecast - newObservation);

      if (this.verbose) {
        console.log(`Previous forecast: ${this.lastForecast.forecast.toFixed(3)}, Actual observation: ${newObservation.toFixed(3)}, Error: ${error.toFixed(3)}`);
      }
    }

    // Add new observation
    this.currentData.push(newObservation);

    // Keep only the most recent observations (rolling window)
    const windowData = this.currentData.slice(-this.windowSize);

    // Fit model on current window
    const model = new ARIMA(this.baseModel.getParams());
    model.fit(windowData);

    // Get next forecast
    const nextForecast = model.forecast(1);
    const forecastResult = {
      forecast: nextForecast.forecast[0],
      lowerBound: nextForecast.lowerBound[0],
      upperBound: nextForecast.upperBound[0]
    };

    // Store this forecast for next error calculation
    this.lastForecast = forecastResult;

    if (this.verbose) {
      console.log(`Next forecast: ${forecastResult.forecast.toFixed(3)}, Window Size: ${windowData.length}`);
    }

    return {
      ...forecastResult,
      error
    };
  }

  setWindowSize(newWindowSize: number): void {
    this.windowSize = newWindowSize;
  }

  getWindowSize(): number {
    return this.windowSize;
  }

  getCurrentData(): number[] {
    return [...this.currentData];
  }

  reset(newInitialData: number[]): void {
    this.currentData = [...newInitialData];
    this.lastForecast = null;
  }
}

export class Adaptive extends ForecastStrategy {
  private stepwiseStrategy: Stepwise;
  private rollingStrategy: RollingWindow;
  private adaptationThreshold: number;
  private errorWindow: number[];
  private maxErrorWindowSize: number;
  private currentStrategy: 'stepwise' | 'rolling';
  private verbose: boolean;

  constructor(
    baseModel: ARIMA,
    initialData: number[],
    options: {
      windowSize?: number;
      adaptationThreshold?: number;
      maxErrorWindowSize?: number;
      verbose?: boolean;
    } = {}
  ) {
    super(baseModel, initialData);

    this.stepwiseStrategy = new Stepwise(baseModel, initialData, {
      refitModel: true,
      verbose: options.verbose
    });

    this.rollingStrategy = new RollingWindow(baseModel, initialData, {
      windowSize: options.windowSize,
      verbose: options.verbose
    });

    this.adaptationThreshold = options.adaptationThreshold ?? 2.0;
    this.maxErrorWindowSize = options.maxErrorWindowSize ?? 10;
    this.errorWindow = [];
    this.currentStrategy = 'stepwise';
    this.verbose = options.verbose ?? false;
  }

  forecast(steps: number): ForecastResult {
    if (this.currentStrategy === 'stepwise') {
      return this.stepwiseStrategy.forecast(steps);
    } else {
      return this.rollingStrategy.forecast(steps);
    }
  }

  forecastWithRealTimeData(actualObservations: number[]): RealTimeForecastResult {
    const results = this.currentStrategy === 'stepwise'
      ? this.stepwiseStrategy.forecastWithRealTimeData(actualObservations)
      : this.rollingStrategy.forecastWithRealTimeData(actualObservations);

    if (results.errors) {
      results.errors.forEach(error => this.addError(error));

      if (this.shouldSwitchStrategy()) {
        this.switchStrategy();
      }
    }

    return results;
  }

  addObservationAndForecast(newObservation: number): { forecast: number; lowerBound: number; upperBound: number; error?: number } {
    const result = this.currentStrategy === 'stepwise'
      ? this.stepwiseStrategy.addObservationAndForecast(newObservation)
      : this.rollingStrategy.addObservationAndForecast(newObservation);

    if (result.error !== undefined) {
      this.addError(result.error);

      if (this.shouldSwitchStrategy()) {
        this.switchStrategy();
      }
    }

    if (this.verbose) {
      const errorText = result.error !== undefined ? `, Error: ${result.error.toFixed(3)}` : '';
      console.log(`Adaptive strategy (${this.currentStrategy}): New observation: ${newObservation.toFixed(3)}, Next forecast: ${result.forecast.toFixed(3)}${errorText}`);
    }

    return result;
  }

  private shouldSwitchStrategy(): boolean {
    if (this.errorWindow.length < this.maxErrorWindowSize) {
      return false;
    }

    const recentAvgError = this.errorWindow.slice(-5).reduce((a, b) => a + b, 0) / 5;
    return recentAvgError > this.adaptationThreshold;
  }

  private addError(error: number): void {
    this.errorWindow.push(error);
    if (this.errorWindow.length > this.maxErrorWindowSize) {
      this.errorWindow.shift();
    }
  }

  private switchStrategy(): void {
    const oldStrategy = this.currentStrategy;
    this.currentStrategy = this.currentStrategy === 'stepwise' ? 'rolling' : 'stepwise';

    if (this.verbose) {
      console.log(`Switching strategy from ${oldStrategy} to ${this.currentStrategy}`);
    }

    this.errorWindow = [];
  }

  getCurrentStrategy(): 'stepwise' | 'rolling' {
    return this.currentStrategy;
  }

  reset(newInitialData: number[]): void {
    this.stepwiseStrategy.reset(newInitialData);
    this.rollingStrategy.reset(newInitialData);
    this.errorWindow = [];
    this.currentStrategy = 'stepwise';
  }
}
