# TS ARIMA Forecast

A TypeScript library for ARIMA (AutoRegressive Integrated Moving Average) time series forecasting with automatic model selection and confidence intervals.

## Features

- 🚀 **Auto ARIMA**: Automatically selects the best ARIMA model parameters
- 📊 **Time Series Forecasting**: Generate accurate forecasts with confidence intervals
- 🔧 **TypeScript Support**: Full TypeScript support with type definitions
- 📈 **Sample Data Generation**: Built-in utilities for testing and examples
- ⚡ **Performance Optimized**: Efficient algorithms for fast computation
- 🔄 **Real-Time Strategies**: Multiple forecasting strategies for live data processing

## Installation

```bash
yarn install ts-arima-forecast
```

## Quick Start

```typescript
import { ARIMA, AutoARIMA, TimeSeriesExample } from 'ts-arima-forecast';

// Generate sample data
const data = TimeSeriesExample.generateSampleData(100);

// Auto-select best ARIMA model
const autoResult = AutoARIMA.findBestARIMA(data);
console.log('Best model:', autoResult.bestParams);

// Fit the model
const model = new ARIMA(autoResult.bestParams);
const fitResult = model.fit(data);

// Generate forecasts
const forecast = model.forecast(10, 0.95);
console.log('Forecast:', forecast.forecast);
console.log('95% CI:', forecast.lowerBound, forecast.upperBound);
```

## API Reference

### ARIMA Class

The main class for ARIMA model fitting and forecasting.

#### Constructor

```typescript
new ARIMA(params: ARIMAParams)
```

- `params`: Object containing `p` (autoregressive order), `d` (differencing order), and `q` (moving average order)

#### Methods

##### `fit(data: number[]): FitResult`

Fits the ARIMA model to the provided time series data.

- `data`: Array of numerical time series values
- Returns: `FitResult` object containing model statistics and fitted values

##### `forecast(steps: number, confidenceLevel?: number): ForecastResult`

Generates forecasts for the specified number of steps ahead.

- `steps`: Number of periods to forecast
- `confidenceLevel`: Confidence level for intervals (default: 0.95)
- Returns: `ForecastResult` with forecast values and confidence bounds

### AutoARIMA Class

Utility class for automatic ARIMA model selection.

#### Methods

##### `findBestARIMA(data: number[], options?: AutoARIMAOptions): AutoARIMAResult`

Automatically finds the best ARIMA model parameters using information criteria.

- `data`: Time series data array
- `options`: Optional configuration for model search
- Returns: `AutoARIMAResult` containing best parameters and model statistics

### TimeSeriesExample Class

Utility class for generating sample time series data.

#### Methods

##### `generateSampleData(length: number, options?: SampleDataOptions): number[]`

Generates synthetic time series data for testing and examples.

- `length`: Number of data points to generate
- `options`: Optional parameters for data generation (trend, seasonality, noise)
- Returns: Array of generated time series values

## Types

### ARIMAParams

```typescript
interface ARIMAParams {
  p: number; // Autoregressive order
  d: number; // Differencing order
  q: number; // Moving average order
}
```

### ForecastResult

```typescript
interface ForecastResult {
  forecast: number[];      // Forecasted values
  lowerBound: number[];   // Lower confidence bounds
  upperBound: number[];   // Upper confidence bounds
  confidenceLevel: number; // Confidence level used
}
```

### AutoARIMAResult

```typescript
interface AutoARIMAResult {
  bestParams: ARIMAParams;
  aic: number;           // Akaike Information Criterion
  bic: number;           // Bayesian Information Criterion
  logLikelihood: number; // Log-likelihood value
}
```

## Examples

### Basic Forecasting

```typescript
import { ARIMA } from 'ts-arima-forecast';

const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const model = new ARIMA({ p: 1, d: 1, q: 1 });

model.fit(data);
const forecast = model.forecast(5);

console.log('Next 5 values:', forecast.forecast);
```

### Custom Confidence Intervals

```typescript
import { ARIMA, AutoARIMA } from 'ts-arima-forecast';

const data = [/* your time series data */];
const autoResult = AutoARIMA.findBestARIMA(data);
const model = new ARIMA(autoResult.bestParams);

model.fit(data);

// 99% confidence intervals
const forecast99 = model.forecast(10, 0.99);
console.log('99% CI:', forecast99.lowerBound, forecast99.upperBound);

// 80% confidence intervals
const forecast80 = model.forecast(10, 0.80);
console.log('80% CI:', forecast80.lowerBound, forecast80.upperBound);
```

### Working with Real Data

```typescript
import { AutoARIMA, ARIMA } from 'ts-arima-forecast';

const salesData = [120, 135, 158, 142, 167, 189, 201, 188, 195, 210];

const autoResult = AutoARIMA.findBestARIMA(salesData);
console.log(`Best model: ARIMA(${autoResult.bestParams.p}, ${autoResult.bestParams.d}, ${autoResult.bestParams.q})`);
console.log(`AIC: ${autoResult.aic}, BIC: ${autoResult.bic}`);

const model = new ARIMA(autoResult.bestParams);
model.fit(salesData);

const forecast = model.forecast(6); // forecast next 6 periods
console.log('6-month forecast:', forecast.forecast);
```

## Real-Time Forecasting Strategies

The library includes three powerful strategies for real-time forecasting scenarios where new observations arrive continuously:

### Stepwise Strategy

The Stepwise strategy continuously updates the model with each new observation, optionally refitting the entire model for maximum accuracy.

```typescript
import { ARIMA, AutoARIMA, Stepwise } from 'ts-arima-forecast';

// 1. Prepare model
const historicalData = [/* your historical time series data */];
const autoResult = AutoARIMA.findBestARIMA(historicalData);
const baseModel = new ARIMA(autoResult.bestParams);
baseModel.fit(historicalData);

// 2. Initialize stepwise strategy
const stepwise = new Stepwise(baseModel, historicalData, {
  refitModel: true,  // refit model with each new observation
  verbose: true      // enable logging
});

// 3. Process new observations one by one
const newObservation = 125.5;
const result = stepwise.addObservationAndForecast(newObservation);

console.log('Next forecast:', result.forecast);
console.log('Confidence interval:', [result.lowerBound, result.upperBound]);
console.log('Forecast error:', result.error); // Error from previous forecast

// Or process multiple observations at once
const newObservations = [125.5, 128.2, 130.1];
const batchResult = stepwise.forecastWithRealTimeData(newObservations);
console.log('Forecasts:', batchResult.forecasts);
console.log('Errors:', batchResult.errors);
```

### Rolling Window Strategy

The Rolling Window strategy maintains a fixed-size window of recent observations, fitting a new model on this window for each forecast.

```typescript
import { RollingWindow } from 'ts-arima-forecast';

// 1. Initialize rolling window strategy
const rollingWindow = new RollingWindow(baseModel, historicalData, {
  windowSize: 50,    // use last 50 observations
  verbose: true
});

// 2. Add new observation and forecast
const result = rollingWindow.addObservationAndForecast(newObservation);
console.log('Next forecast:', result.forecast);
console.log('Window size:', rollingWindow.getWindowSize());

// 3. Adjust window size (if you want to)
rollingWindow.setWindowSize(30);

// 4. Process batch of observations
const batchResult = rollingWindow.forecastWithRealTimeData(newObservations);
console.log('Average error:', 
  batchResult.errors!.reduce((a, b) => a + b, 0) / batchResult.errors!.length
);
```

### Adaptive Strategy

The Adaptive strategy automatically switches between Stepwise and Rolling Window approaches based on forecast performance.

```typescript
import { Adaptive } from 'ts-arima-forecast';

// 1. Initialize adaptive strategy
const adaptive = new Adaptive(baseModel, historicalData, {
  windowSize: 40,              // Window size for rolling strategy
  adaptationThreshold: 2.5,    // Error threshold for switching
  maxErrorWindowSize: 10,      // Number of errors to track
  verbose: true
});

// The strategy will automatically switch based on performance
const result = adaptive.addObservationAndForecast(newObservation);
console.log('Current strategy:', adaptive.getCurrentStrategy()); // 'stepwise' or 'rolling'
console.log('Next forecast:', result.forecast);

// 2. Process multiple observations with automatic adaptation
const batchResult = adaptive.forecastWithRealTimeData(newObservations);
console.log('Final strategy:', adaptive.getCurrentStrategy());
```

### Strategy Comparison Example

Compare all three strategies to find the best performer for your data:

```typescript
import { Stepwise, RollingWindow, Adaptive } from 'ts-arima-forecast';

// Prepare test data
const trainData = historicalData.slice(0, 80);
const testData = historicalData.slice(80);

// Initialize all strategies
const strategies = {
  stepwise: new Stepwise(baseModel, trainData, { refitModel: true }),
  rollingWindow: new RollingWindow(baseModel, trainData, { windowSize: 40 }),
  adaptive: new Adaptive(baseModel, trainData, { 
    windowSize: 40, 
    adaptationThreshold: 2.0 
  })
};

// Compare performance
const results: { [key: string]: number } = {};

for (const [name, strategy] of Object.entries(strategies)) {
  const result = strategy.forecastWithRealTimeData(testData);
  const avgError = result.errors!.reduce((a, b) => a + b, 0) / result.errors!.length;
  results[name] = avgError;
  console.log(`${name}: Average error = ${avgError.toFixed(3)}`);
}

// Find best strategy
const bestStrategy = Object.entries(results).reduce((a, b) => 
  results[a[0]] < results[b[0]] ? a : b
);
console.log(`Best strategy: ${bestStrategy[0]} (error: ${bestStrategy[1].toFixed(3)})`);
```

### Real-Time Data Pipeline Example

Here's how to set up a complete real-time forecasting pipeline:

```typescript
import { ARIMA, AutoARIMA, Adaptive } from 'ts-arima-forecast';

class RealTimeForecastPipeline {
  private strategy: Adaptive;
  
  constructor(historicalData: number[]) {
    // Auto-select best model
    const autoResult = AutoARIMA.findBestARIMA(historicalData);
    const baseModel = new ARIMA(autoResult.bestParams);
    baseModel.fit(historicalData);
    
    // Initialize adaptive strategy
    this.strategy = new Adaptive(baseModel, historicalData, {
      windowSize: 50,
      adaptationThreshold: 3.0,
      verbose: false
    });
  }
  
  // Process new data point
  processNewData(value: number): {
    forecast: number;
    confidence: [number, number];
    error?: number;
    strategy: string;
  } {
    const result = this.strategy.addObservationAndForecast(value);
    
    return {
      forecast: result.forecast,
      confidence: [result.lowerBound, result.upperBound],
      error: result.error,
      strategy: this.strategy.getCurrentStrategy()
    };
  }
  
  // Get current model state
  getStatus() {
    return {
      currentStrategy: this.strategy.getCurrentStrategy(),
      dataPoints: this.strategy.getCurrentData?.()?.length || 0
    };
  }
}

// Usage
const pipeline = new RealTimeForecastPipeline(historicalData);

// Simulate real-time data arrival
setInterval(() => {
  const newValue = getNewDataPoint(); // Your data source
  const result = pipeline.processNewData(newValue);
  
  console.log(`Forecast: ${result.forecast.toFixed(2)}`);
  console.log(`Strategy: ${result.strategy}`);
  if (result.error) {
    console.log(`Previous error: ${result.error.toFixed(2)}`);
  }
}, 1000); // Process new data every second
```

### Strategy Selection Guidelines

**Use Stepwise Strategy when:**
- You have sufficient computational resources for model refitting
- Data patterns change gradually over time
- You need maximum forecast accuracy
- Model interpretability is important

**Use Rolling Window Strategy when:**
- You need consistent performance with limited resources
- Data has recurring patterns within a fixed window
- You want to limit the influence of very old data
- Processing speed is critical

**Use Adaptive Strategy when:**
- Data patterns are unpredictable
- You want automatic optimization
- You're unsure which approach works best
- You need robust performance across different scenarios

## Requirements

- Node.js >= 14
- TypeScript >= 4.0 (for TypeScript projects)

## License

GNU Lesser General Public License v2.1

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please file an issue on the GitHub repository.